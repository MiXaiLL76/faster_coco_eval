#include <time.h>

#include <algorithm>
#include <cstdint>
#include <numeric>

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

        try {
                if (ann_dict.contains("id")) {
                        id = ann_dict["id"].cast<uint64_t>();
                }
        } catch (const std::exception&) {
        }

        try {
                if (ann_dict.contains("score")) {
                        score = ann_dict["score"].cast<double>();
                }
        } catch (const std::exception&) {
        }

        try {
                if (ann_dict.contains("area")) {
                        area = ann_dict["area"].cast<double>();
                }
        } catch (const std::exception&) {
        }

        try {
                if (ann_dict.contains("is_crowd")) {
                        is_crowd = ann_dict["is_crowd"].cast<bool>();
                } else if (ann_dict.contains("iscrowd")) {
                        is_crowd = ann_dict["iscrowd"].cast<bool>();
                }
        } catch (const std::exception&) {
        }

        try {
                if (ann_dict.contains("ignore")) {
                        ignore = ann_dict["ignore"].cast<bool>();
                }
        } catch (const std::exception&) {
        }

        try {
                if (ann_dict.contains("lvis_mark")) {
                        lvis_mark = ann_dict["lvis_mark"].cast<bool>();
                }
        } catch (const std::exception&) {
        }

        // Construct and return the annotation.
        return InstanceAnnotation(id, score, area, is_crowd, ignore, lvis_mark);
}

// Get C++ annotation objects with caching for performance
std::vector<InstanceAnnotation> LightweightDataset::get_cpp_annotations(
    double img_id, double cat_id) const {
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
                cpp_cache[key] = result;
                return result;
        } else {
                return {};
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
                                      const bool& useCats) {
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
