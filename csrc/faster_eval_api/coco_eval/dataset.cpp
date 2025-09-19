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
        const std::pair<int64_t, int64_t> key{static_cast<int64_t>(img_id),
                                              static_cast<int64_t>(cat_id)};

        // Convert and store in JSON format for memory optimization testing
        json json_ann = ann;  // py::dict → json conversion
        data[key].emplace_back(std::move(json_ann));
}

// Removes all stored annotations and frees internal memory used by the data
// container.
void Dataset::clean() {
        data.clear();

        // Optionally reclaim memory by swapping with empty maps (C++17 idiom)
        std::unordered_map<std::pair<int64_t, int64_t>, std::vector<json>,
                           hash_pair>()
            .swap(data);
}

// Get dataset size (number of (img_id, cat_id) pairs with annotations).
size_t Dataset::size() const { return data.size(); }

// Serializes the Dataset into a Python tuple for pickling support.
// Stores size and data as string for memory efficient transfer.
py::tuple Dataset::make_tuple() const {
        // Create JSON object with all data
        json serialized_data;
        for (const auto &kv : data) {
                // Convert key pair to string for JSON compatibility
                std::string key_str = std::to_string(kv.first.first) + "," +
                                      std::to_string(kv.first.second);
                serialized_data[key_str] = kv.second;
        }

        // Convert to string
        std::string json_string = serialized_data.dump();

        // Return as Python tuple (size, json_string).
        return py::make_tuple(static_cast<int>(data.size()), json_string);
}

// Loads the Dataset state from a Python tuple (typically for unpickling).
// The tuple must contain two elements: size (int) and json_string (str).
// Throws a runtime error if the state is invalid.
void Dataset::load_tuple(py::tuple pickle_data) {
        if (pickle_data.size() != 2)
                throw std::runtime_error(
                    "Invalid state! Tuple must have 2 elements.");

        // Get size and json string from tuple
        int expected_size = pickle_data[0].cast<int>();
        std::string json_string = pickle_data[1].cast<std::string>();

        // Parse JSON from string
        json serialized_data = json::parse(json_string);

        // Clear existing data and reserve memory for efficiency.
        data.clear();
        data.reserve(expected_size);

        // Reconstruct data from parsed JSON
        for (const auto &item : serialized_data.items()) {
                const std::string &key_str = item.key();
                const json &value = item.value();

                // Parse key string back to pair<int64_t, int64_t>
                size_t comma_pos = key_str.find(',');
                if (comma_pos != std::string::npos) {
                        int64_t img_id =
                            std::stoll(key_str.substr(0, comma_pos));
                        int64_t cat_id =
                            std::stoll(key_str.substr(comma_pos + 1));

                        data.emplace(std::make_pair(img_id, cat_id),
                                     value.get<std::vector<json>>());
                }
        }
}

// Returns a vector of Python dictionaries for a given (img_id, cat_id) pair.
// If the pair is not found, returns an empty vector.
// Works directly with data storage for memory optimization.
std::vector<py::dict> Dataset::get(double img_id, double cat_id) {
        const std::pair<int64_t, int64_t> key(static_cast<int64_t>(img_id),
                                              static_cast<int64_t>(cat_id));
        auto it = data.find(key);
        if (it != data.end()) {
                // Convert json → py::dict for return
                std::vector<py::dict> result;
                result.reserve(it->second.size());

                for (const auto &json_ann : it->second) {
                        // Convert json → py::dict
                        py::dict converted_dict = json_ann;
                        result.emplace_back(std::move(converted_dict));
                }

                return result;
        } else {
                // Return empty vector if key is not found.
                return {};
        }
}

// Parses a JSON annotation into an InstanceAnnotation object.
// Extracts fields by name directly from JSON for better performance.
InstanceAnnotation parseInstanceAnnotation(const json &ann) {
        uint64_t id = 0;
        double score = 0.0;
        double area = 0.0;
        bool is_crowd = false;
        bool ignore = false;
        bool lvis_mark = false;

        // Extract values directly from JSON with flexible type handling
        if (ann.contains("id")) {
                if (ann["id"].is_number_integer()) {
                        id = ann["id"].get<uint64_t>();
                } else if (ann["id"].is_number_float()) {
                        id = static_cast<uint64_t>(ann["id"].get<double>());
                } else if (ann["id"].is_string()) {
                        id = std::stoull(ann["id"].get<std::string>());
                }
        }
        if (ann.contains("score")) {
                if (ann["score"].is_number()) {
                        score = ann["score"].get<double>();
                } else if (ann["score"].is_string()) {
                        score = std::stod(ann["score"].get<std::string>());
                }
        }
        if (ann.contains("area")) {
                if (ann["area"].is_number()) {
                        area = ann["area"].get<double>();
                } else if (ann["area"].is_string()) {
                        area = std::stod(ann["area"].get<std::string>());
                }
        }
        if (ann.contains("is_crowd")) {
                if (ann["is_crowd"].is_boolean()) {
                        is_crowd = ann["is_crowd"].get<bool>();
                } else if (ann["is_crowd"].is_number()) {
                        is_crowd = ann["is_crowd"].get<int>() != 0;
                }
        } else if (ann.contains("iscrowd")) {
                if (ann["iscrowd"].is_boolean()) {
                        is_crowd = ann["iscrowd"].get<bool>();
                } else if (ann["iscrowd"].is_number()) {
                        is_crowd = ann["iscrowd"].get<int>() != 0;
                }
        }
        if (ann.contains("ignore")) {
                if (ann["ignore"].is_boolean()) {
                        ignore = ann["ignore"].get<bool>();
                } else if (ann["ignore"].is_number()) {
                        ignore = ann["ignore"].get<int>() != 0;
                }
        }
        if (ann.contains("lvis_mark")) {
                if (ann["lvis_mark"].is_boolean()) {
                        lvis_mark = ann["lvis_mark"].get<bool>();
                } else if (ann["lvis_mark"].is_number()) {
                        lvis_mark = ann["lvis_mark"].get<int>() != 0;
                }
        }

        // Construct and return the annotation.
        return InstanceAnnotation(id, score, area, is_crowd, ignore, lvis_mark);
}

// Returns a vector of InstanceAnnotation objects for a given (img_id, cat_id)
// pair. Works directly with data for efficiency.
std::vector<InstanceAnnotation> Dataset::get_cpp_annotations(double img_id,
                                                             double cat_id) {
        const std::pair<int64_t, int64_t> key(static_cast<int64_t>(img_id),
                                              static_cast<int64_t>(cat_id));
        auto it = data.find(key);
        if (it != data.end()) {
                std::vector<InstanceAnnotation> result;
                result.reserve(it->second.size());

                // Convert each JSON annotation directly to InstanceAnnotation.
                for (const auto &json_ann : it->second) {
                        result.emplace_back(parseInstanceAnnotation(json_ann));
                }
                return result;
        } else {
                return {};
        }
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
