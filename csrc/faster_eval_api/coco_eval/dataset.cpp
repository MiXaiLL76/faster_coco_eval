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
// Appends an annotation to the dataset for a specific (img_id, cat_id) pair.
// Uses emplace_back for efficient insertion. Accepts ann as const reference to
// avoid unnecessary copying.
void Dataset::append(double img_id, double cat_id, const py::dict &ann) {
        // Use emplace_back to construct py::dict in-place for efficiency.
        data[{static_cast<int64_t>(img_id), static_cast<int64_t>(cat_id)}]
            .emplace_back(ann);
}

// Removes all stored annotations and frees internal memory used by the data
// container.
void Dataset::clean() {
        data.clear();
        // Optionally reclaim memory by swapping with an empty map (C++17 idiom)
        std::unordered_map<std::pair<int64_t, int64_t>, std::vector<py::dict>,
                           hash_pair>()
            .swap(data);
}

// Get dataset size (number of (img_id, cat_id) pairs with annotations).
size_t Dataset::size() const { return data.size(); }

// Serializes the Dataset into a Python tuple for pickling support.
// Stores the keys and values as two separate vectors for efficient transfer to
// Python.
py::tuple Dataset::make_tuple() const {
        // Preallocate storage for better memory efficiency.
        std::vector<std::pair<int64_t, int64_t>> keys;
        std::vector<std::vector<py::dict>> values;
        keys.reserve(data.size());
        values.reserve(data.size());

        // Iterate over all key-value pairs and collect them.
        for (const auto &kv : data) {
                keys.push_back(kv.first);
                values.push_back(kv.second);
        }
        // Return as Python tuple (keys, values).
        return py::make_tuple(keys, values);
}

// Loads the Dataset state from a Python tuple (typically for unpickling).
// The tuple must contain two elements: a vector of keys and a vector of value
// vectors. Throws a runtime error if the state is invalid.
void Dataset::load_tuple(py::tuple pickle_data) {
        if (pickle_data.size() != 2)
                throw std::runtime_error(
                    "Invalid state! Tuple must have 2 elements.");

        // Cast Python objects to C++ vectors.
        std::vector<std::pair<int64_t, int64_t>> keys =
            pickle_data[0].cast<std::vector<std::pair<int64_t, int64_t>>>();
        std::vector<std::vector<py::dict>> values =
            pickle_data[1].cast<std::vector<std::vector<py::dict>>>();

        if (keys.size() != values.size())
                throw std::runtime_error(
                    "Invalid state! Keys and values vectors must have the same "
                    "size.");

        // Clear existing data and reserve memory for efficiency.
        data.clear();
        data.reserve(keys.size());

        // Insert each key-value pair using move semantics for optimal
        // performance.
        for (size_t i = 0; i < keys.size(); ++i) {
                data.emplace(std::move(keys[i]), std::move(values[i]));
        }
}

// Returns a vector of Python dictionaries for a given (img_id, cat_id) pair.
// If the pair is not found, returns an empty vector.
// Uses find() to avoid unnecessary construction or lookup.
std::vector<py::dict> Dataset::get(double img_id, double cat_id) {
        const std::pair<int64_t, int64_t> key(static_cast<int64_t>(img_id),
                                              static_cast<int64_t>(cat_id));
        auto it = data.find(key);
        if (it != data.end()) {
                return it->second;
        } else {
                // Return empty vector if key is not found.
                return {};
        }
}

// Parses a py::dict annotation into an InstanceAnnotation object.
// Extracts fields by name and casts to appropriate type.
// Optimized for C++17: uses const references, avoids unnecessary string copies.
InstanceAnnotation parseInstanceAnnotation(const py::dict &ann) {
        uint64_t id = 0;
        double score = 0.0;
        double area = 0.0;
        bool is_crowd = false;
        bool ignore = false;
        bool lvis_mark = false;

        // Iterate using const reference for efficiency.
        for (const auto &item : ann) {
                const auto &key_obj = item.first;
                const auto &val_obj = item.second;

                std::string key = key_obj.cast<std::string>();

                if (key == "id") {
                        id = val_obj.cast<uint64_t>();
                } else if (key == "score") {
                        score = val_obj.cast<double>();
                } else if (key == "area") {
                        area = val_obj.cast<double>();
                } else if (key == "is_crowd" || key == "iscrowd") {
                        is_crowd = val_obj.cast<bool>();
                } else if (key == "ignore") {
                        ignore = val_obj.cast<bool>();
                } else if (key == "lvis_mark") {
                        lvis_mark = val_obj.cast<bool>();
                }
        }
        // Construct and return the annotation.
        return InstanceAnnotation(id, score, area, is_crowd, ignore, lvis_mark);
}

// Returns a vector of InstanceAnnotation objects for a given (img_id, cat_id)
// pair. Uses reserve() for performance and emplace_back for efficient
// insertion.
std::vector<InstanceAnnotation> Dataset::get_cpp_annotations(double img_id,
                                                             double cat_id) {
        std::vector<py::dict> anns = get(img_id, cat_id);
        std::vector<InstanceAnnotation> result;
        result.reserve(anns.size());  // Reserve space to avoid reallocations.

        // Convert each py::dict annotation to InstanceAnnotation.
        for (const auto &ann : anns) {
                result.emplace_back(parseInstanceAnnotation(ann));
        }
        return result;
}

// Returns all InstanceAnnotations for each combination of img_ids and cat_ids.
// If useCats is false, all category results for an image are merged into a
// single vector. Optimized for better memory management and clarity.
std::vector<std::vector<std::vector<InstanceAnnotation>>>
Dataset::get_cpp_instances(const std::vector<double> &img_ids,
                           const std::vector<double> &cat_ids,
                           const bool &useCats) {
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

// Returns all py::dict annotations for each combination of img_ids and cat_ids.
// If useCats is false, all category results for an image are merged into a
// single vector. Optimized for better memory management and clarity.
std::vector<std::vector<std::vector<py::dict>>> Dataset::get_instances(
    const std::vector<double> &img_ids, const std::vector<double> &cat_ids,
    const bool &useCats) {
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
