#include <time.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <type_traits>

// clang-format off
#include "cocoeval.h"
#include "dataset.h"
// clang-format on

using namespace pybind11::literals;

namespace coco_eval {

namespace COCOeval {

// Store reference to annotation instead of copying data
void LightweightDataset::append_ref(double img_id, double cat_id,
                                    py::object ann_ref) {
        const std::pair<int64_t, int64_t> key{static_cast<int64_t>(img_id),
                                              static_cast<int64_t>(cat_id)};
        annotation_refs[key].emplace_back(ann_ref);
}

// Store a whole annotation sequence in one call.
//
// The per-annotation append_ref binding costs roughly 287ns of pybind dispatch
// per call, which dominates ingest for detection-sized inputs. Looping here
// instead pays that cost once. The returned set of (image_id, category_id)
// tuples is what the caller would otherwise rebuild in a second Python loop.
py::set LightweightDataset::append_batch(const py::sequence& annotations,
                                         bool skip_dropped) {
        py::set pairs;

        const py::str image_key("image_id");
        const py::str category_key("category_id");
        const py::str drop_key("drop");

        for (const py::handle& item : annotations) {
                const py::dict ann = py::reinterpret_borrow<py::dict>(item);

                if (skip_dropped) {
                        // Mirror Python's ``not ann.get("drop", False)``:
                        // truthiness, not a bool cast. Annotations come from
                        // user data, so "drop" may hold any object; casting
                        // would raise where the original loop simply evaluated
                        // it.
                        const py::object drop_obj =
                            ann.contains(drop_key)
                                ? py::reinterpret_borrow<py::object>(
                                      ann[drop_key])
                                : py::none();
                        const int is_true = PyObject_IsTrue(drop_obj.ptr());
                        if (is_true == -1) {
                                throw py::error_already_set();
                        }
                        if (is_true == 1) {
                                continue;
                        }
                }

                const py::object img_obj = ann[image_key];
                const py::object cat_obj = ann[category_key];

                const std::pair<int64_t, int64_t> key{
                    static_cast<int64_t>(img_obj.cast<double>()),
                    static_cast<int64_t>(cat_obj.cast<double>())};
                annotation_refs[key].emplace_back(
                    py::reinterpret_borrow<py::object>(item));

                // Carry the original id objects, not the narrowed int64 copies,
                // so the caller's pair set stays identical to the Python
                // loop's.
                pairs.add(py::make_tuple(img_obj, cat_obj));
        }

        return pairs;
}

// Remove all stored references and clear cache
void LightweightDataset::clean() {
        annotation_refs.clear();
        cpp_cache.clear();

        // Reclaim memory by swapping with empty containers
        std::unordered_map<std::pair<int64_t, int64_t>, std::vector<py::object>,
                           hash_pair>()
            .swap(annotation_refs);
        std::unordered_map<std::pair<int64_t, int64_t>,
                           std::vector<InstanceAnnotation>, hash_pair>()
            .swap(cpp_cache);
}

// Get dataset size (number of (img_id, cat_id) pairs with annotations)
size_t LightweightDataset::size() const { return annotation_refs.size(); }

// Serialize dataset contents to a tuple for pickle support
py::tuple LightweightDataset::make_tuple() const {
        // Create a list of (img_id, cat_id, annotation_list) tuples
        py::list serialized_data;
        for (const auto& kv : annotation_refs) {
                auto key = kv.first;
                auto ann_list = kv.second;

                py::list py_ann_list;
                for (const auto& ann : ann_list) {
                        py_ann_list.append(ann);
                }

                serialized_data.append(py::make_tuple(
                    static_cast<double>(key.first),
                    static_cast<double>(key.second), py_ann_list));
        }

        return py::make_tuple(static_cast<int>(annotation_refs.size()),
                              serialized_data);
}

// Load dataset state from a Python tuple (for unpickling)
void LightweightDataset::load_tuple(py::tuple pickle_data) {
        if (pickle_data.size() != 2)
                throw std::runtime_error(
                    "Invalid state! Tuple must have 2 elements.");

        // Get size and data from tuple
        int expected_size = pickle_data[0].cast<int>();
        py::list serialized_data = pickle_data[1].cast<py::list>();

        // Clear existing data and reserve memory
        annotation_refs.clear();
        cpp_cache.clear();
        annotation_refs.reserve(expected_size);

        // Reconstruct data from serialized list
        for (auto item : serialized_data) {
                py::tuple entry = item.cast<py::tuple>();
                if (entry.size() != 3) continue;

                double img_id = entry[0].cast<double>();
                double cat_id = entry[1].cast<double>();
                py::list ann_list = entry[2].cast<py::list>();

                std::pair<int64_t, int64_t> key{static_cast<int64_t>(img_id),
                                                static_cast<int64_t>(cat_id)};

                std::vector<py::object> annotations;
                for (auto ann : ann_list) {
                        annotations.emplace_back(
                            py::reinterpret_borrow<py::object>(ann));
                }

                annotation_refs[key] = std::move(annotations);
        }
}

// Get all Python dict annotations for a given image/category pair
std::vector<py::dict> LightweightDataset::get(double img_id, double cat_id) {
        const std::pair<int64_t, int64_t> key(static_cast<int64_t>(img_id),
                                              static_cast<int64_t>(cat_id));
        auto it = annotation_refs.find(key);
        if (it != annotation_refs.end()) {
                std::vector<py::dict> result;
                result.reserve(it->second.size());

                for (const auto& py_ann : it->second) {
                        // Convert py::object to py::dict
                        result.emplace_back(py_ann.cast<py::dict>());
                }

                return result;
        } else {
                return {};
        }
}

namespace {

// Annotation field names, interned once for the process lifetime.
//
// Passing a C string literal to dict::contains or operator[] makes pybind11
// build a fresh Python str for every lookup: allocate, decode UTF-8, hash with
// siphash24 because a new str has no cached hash, then free. That repeated for
// seven fields on every annotation dominated annotation parsing. An interned
// str is built once and carries its hash, so lookups become a plain
// PyDict_GetItem probe.
struct AnnotationKeys {
        PyObject* id;
        PyObject* score;
        PyObject* area;
        PyObject* is_crowd;
        PyObject* iscrowd;
        PyObject* ignore;
        PyObject* lvis_mark;

        AnnotationKeys()
            : id(PyUnicode_InternFromString("id")),
              score(PyUnicode_InternFromString("score")),
              area(PyUnicode_InternFromString("area")),
              is_crowd(PyUnicode_InternFromString("is_crowd")),
              iscrowd(PyUnicode_InternFromString("iscrowd")),
              ignore(PyUnicode_InternFromString("ignore")),
              lvis_mark(PyUnicode_InternFromString("lvis_mark")) {}

        // Owns seven raw PyObject* and defines a destructor: forbid copy and
        // move so an accidental by-value use cannot double Py_XDECREF the
        // interned strings. The single instance is reached via
        // annotation_keys().
        AnnotationKeys(const AnnotationKeys&) = delete;
        AnnotationKeys& operator=(const AnnotationKeys&) = delete;
        AnnotationKeys(AnnotationKeys&&) = delete;
        AnnotationKeys& operator=(AnnotationKeys&&) = delete;

        bool complete() const {
                return id != nullptr && score != nullptr && area != nullptr &&
                       is_crowd != nullptr && iscrowd != nullptr &&
                       ignore != nullptr && lvis_mark != nullptr;
        }

        ~AnnotationKeys() {
                Py_XDECREF(id);
                Py_XDECREF(score);
                Py_XDECREF(area);
                Py_XDECREF(is_crowd);
                Py_XDECREF(iscrowd);
                Py_XDECREF(ignore);
                Py_XDECREF(lvis_mark);
        }
};

// A copy or move would run ~AnnotationKeys twice on the same interned
// PyObject*, double Py_XDECREF them and corrupt interpreter refcounts.
static_assert(!std::is_copy_constructible<AnnotationKeys>::value &&
                  !std::is_move_constructible<AnnotationKeys>::value &&
                  !std::is_copy_assignable<AnnotationKeys>::value &&
                  !std::is_move_assignable<AnnotationKeys>::value,
              "AnnotationKeys must not be copyable or movable");

// Deliberately leaked: interpreter teardown may run static destructors without
// the GIL, and releasing Python objects there is undefined behaviour. The keys
// are needed for as long as the module is usable, so never freeing them costs
// seven small strings and removes the shutdown hazard.
const AnnotationKeys& annotation_keys() {
        static const AnnotationKeys* keys = [] {
                auto* candidate = new AnnotationKeys();
                if (candidate->complete()) {
                        return candidate;
                }

                delete candidate;
                if (PyErr_Occurred()) {
                        throw py::error_already_set();
                }
                throw std::runtime_error("Failed to intern annotation keys.");
        }();
        return *keys;
}

enum class DictLookupStatus { found, missing, error };

// Borrowed lookup with an explicit absent-versus-error outcome.
//
// Annotation parsing remains best-effort, matching the previous per-field
// exception handling. The status prevents a failed preferred crowd-key lookup
// from being treated as absent and falling back to the legacy spelling.
//
// PyDict_GetItemWithError is a C-level slot read: it assumes annotations are
// plain dicts (as produced from COCO/LVIS JSON) and does not honour a dict
// subclass that overrides __getitem__/__missing__, unlike the previous
// contains()+operator[] path.
struct DictLookup {
        PyObject* value;
        DictLookupStatus status;
};

inline DictLookup dict_get(const py::dict& mapping, PyObject* key) {
        PyObject* value = PyDict_GetItemWithError(mapping.ptr(), key);
        if (value != nullptr) {
                return {value, DictLookupStatus::found};
        }
        if (value == nullptr && PyErr_Occurred()) {
                PyErr_Clear();
                return {nullptr, DictLookupStatus::error};
        }
        return {nullptr, DictLookupStatus::missing};
}

}  // namespace

// Helper method to convert py::object to InstanceAnnotation
InstanceAnnotation LightweightDataset::parse_py_annotation(
    const py::object& ann) const {
        uint64_t id = 0;
        double score = 0.0;
        double area = 0.0;
        bool is_crowd = false;
        bool ignore = false;
        bool lvis_mark = false;

        // Extract values from Python dict with safe type handling
        py::dict ann_dict = ann.cast<py::dict>();
        const AnnotationKeys& keys = annotation_keys();

        // Each field keeps its own try/catch so that one unconvertible value
        // leaves that field at its default without discarding the others.
        if (const DictLookup lookup = dict_get(ann_dict, keys.id);
            lookup.value != nullptr) {
                try {
                        id = py::handle(lookup.value).cast<uint64_t>();
                } catch (const std::exception&) {
                }
        }

        if (const DictLookup lookup = dict_get(ann_dict, keys.score);
            lookup.value != nullptr) {
                try {
                        score = py::handle(lookup.value).cast<double>();
                } catch (const std::exception&) {
                }
        }

        if (const DictLookup lookup = dict_get(ann_dict, keys.area);
            lookup.value != nullptr) {
                try {
                        area = py::handle(lookup.value).cast<double>();
                } catch (const std::exception&) {
                }
        }

        // "is_crowd" wins when present; "iscrowd" is only consulted if the
        // preferred spelling is absent, matching the original else-if.
        DictLookup crowd_lookup = dict_get(ann_dict, keys.is_crowd);
        if (crowd_lookup.status == DictLookupStatus::missing) {
                crowd_lookup = dict_get(ann_dict, keys.iscrowd);
        }
        if (crowd_lookup.value != nullptr) {
                try {
                        is_crowd = py::handle(crowd_lookup.value).cast<bool>();
                } catch (const std::exception&) {
                }
        }

        if (const DictLookup lookup = dict_get(ann_dict, keys.ignore);
            lookup.value != nullptr) {
                try {
                        ignore = py::handle(lookup.value).cast<bool>();
                } catch (const std::exception&) {
                }
        }

        if (const DictLookup lookup = dict_get(ann_dict, keys.lvis_mark);
            lookup.value != nullptr) {
                try {
                        lvis_mark = py::handle(lookup.value).cast<bool>();
                } catch (const std::exception&) {
                }
        }

        // Construct and return the annotation.
        return InstanceAnnotation(id, score, area, is_crowd, ignore, lvis_mark);
}

// Get C++ annotation objects with caching for performance
const std::vector<InstanceAnnotation>& LightweightDataset::get_cpp_annotations(
    double img_id, double cat_id) const {
        static const std::vector<InstanceAnnotation> kEmpty;
        const std::pair<int64_t, int64_t> key(static_cast<int64_t>(img_id),
                                              static_cast<int64_t>(cat_id));

        // Check cache first
        auto cache_it = cpp_cache.find(key);
        if (cache_it != cpp_cache.end()) {
                return cache_it->second;
        }

        // If not in cache, get from annotation_refs and convert
        auto it = annotation_refs.find(key);
        if (it != annotation_refs.end()) {
                std::vector<InstanceAnnotation> result;
                result.reserve(it->second.size());

                // Convert each Python annotation to InstanceAnnotation
                for (const auto& py_ann : it->second) {
                        result.emplace_back(parse_py_annotation(py_ann));
                }

                // Cache the result for future use
                auto inserted = cpp_cache.emplace(key, std::move(result));
                return inserted.first->second;
        } else {
                return kEmpty;
        }
}

// Clear cache entry for specific (img_id, cat_id) to free memory
void LightweightDataset::clear_cache_entry(double img_id, double cat_id) const {
        const std::pair<int64_t, int64_t> key(static_cast<int64_t>(img_id),
                                              static_cast<int64_t>(cat_id));
        cpp_cache.erase(key);
}

// Get all C++ annotation objects for provided img_ids and cat_ids
std::vector<std::vector<std::vector<InstanceAnnotation>>>
LightweightDataset::get_cpp_instances(const std::vector<double>& img_ids,
                                      const std::vector<double>& cat_ids,
                                      const bool& useCats) const {
        std::vector<std::vector<std::vector<InstanceAnnotation>>> result;
        result.reserve(img_ids.size());  // Reserve space for image indices

        for (size_t i = 0; i < img_ids.size(); ++i) {
                int64_t img_id = static_cast<int64_t>(img_ids[i]);
                if (useCats) {
                        std::vector<std::vector<InstanceAnnotation>>
                            cat_results;
                        cat_results.reserve(
                            cat_ids.size());  // Reserve space for categories

                        for (size_t j = 0; j < cat_ids.size(); ++j) {
                                int64_t cat_id =
                                    static_cast<int64_t>(cat_ids[j]);
                                cat_results.emplace_back(
                                    get_cpp_annotations(img_id, cat_id));
                        }
                        result.emplace_back(std::move(cat_results));
                } else {
                        // Single vector to merge all categories for this image
                        std::vector<InstanceAnnotation> merged;
                        for (size_t j = 0; j < cat_ids.size(); ++j) {
                                int64_t cat_id =
                                    static_cast<int64_t>(cat_ids[j]);
                                std::vector<InstanceAnnotation> anns =
                                    get_cpp_annotations(img_id, cat_id);
                                merged.insert(
                                    merged.end(),
                                    std::make_move_iterator(anns.begin()),
                                    std::make_move_iterator(anns.end()));
                        }
                        // Wrap merged vector in an outer vector for consistency
                        result.emplace_back(1, std::move(merged));
                }
        }
        return result;
}

// Get all Python dict annotations for provided img_ids and cat_ids
std::vector<std::vector<std::vector<py::dict>>>
LightweightDataset::get_instances(const std::vector<double>& img_ids,
                                  const std::vector<double>& cat_ids,
                                  const bool& useCats) {
        std::vector<std::vector<std::vector<py::dict>>> result;
        result.reserve(img_ids.size());  // Reserve space for images

        for (size_t i = 0; i < img_ids.size(); ++i) {
                int64_t img_id = static_cast<int64_t>(img_ids[i]);

                if (useCats) {
                        std::vector<std::vector<py::dict>> cat_results;
                        cat_results.reserve(
                            cat_ids.size());  // Reserve space for categories

                        for (size_t j = 0; j < cat_ids.size(); ++j) {
                                int64_t cat_id =
                                    static_cast<int64_t>(cat_ids[j]);
                                cat_results.emplace_back(get(img_id, cat_id));
                        }
                        result.emplace_back(std::move(cat_results));
                } else {
                        // Single vector to merge all categories for this image
                        std::vector<py::dict> merged;
                        for (size_t j = 0; j < cat_ids.size(); ++j) {
                                int64_t cat_id =
                                    static_cast<int64_t>(cat_ids[j]);

                                std::vector<py::dict> anns =
                                    get(img_id, cat_id);
                                merged.insert(
                                    merged.end(),
                                    std::make_move_iterator(anns.begin()),
                                    std::make_move_iterator(anns.end()));
                        }
                        // Wrap merged vector in an outer vector for consistency
                        result.emplace_back(1, std::move(merged));
                }
        }
        return result;
}

}  // namespace COCOeval

}  // namespace coco_eval
