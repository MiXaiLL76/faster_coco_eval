#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "types.h"

namespace py = pybind11;

namespace coco_eval {

namespace COCOeval {
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct hash_pair {
        std::size_t operator()(const std::pair<int64_t, int64_t>& p) const {
                std::size_t h = 0;
                hash_combine(h, p.first);
                hash_combine(h, p.second);
                return h;
        }
};

class LightweightDataset {
       public:
        LightweightDataset() {
                // Reserve initial space to reduce rehashing
                annotation_refs.reserve(8192);
        }

        // Store reference to annotation instead of copying data
        void append_ref(double img_id, double cat_id, py::object ann_ref);

        // Remove all stored references and clear cache
        void clean();

        // Get dataset size (number of (img_id, cat_id) pairs with annotations)
        size_t size() const;

        // Pickle support: Serialize dataset contents to a tuple
        py::tuple make_tuple() const;

        // Pickle support: Load dataset contents from a tuple
        void load_tuple(py::tuple pickle_data);

        // Get all Python dict annotations for a given image/category pair
        std::vector<py::dict> get(double img_id, double cat_id);

        // Get C++ annotation objects with caching for performance
        std::vector<InstanceAnnotation> get_cpp_annotations(
            double img_id, double cat_id) const;

        // Clear cache entry for specific (img_id, cat_id) to free memory
        void clear_cache_entry(double img_id, double cat_id) const;

        // Get all C++ annotation objects for provided img_ids and cat_ids
        std::vector<std::vector<std::vector<InstanceAnnotation>>>
        get_cpp_instances(const std::vector<double>& img_ids,
                          const std::vector<double>& cat_ids,
                          const bool& useCats);

        // Get all Python dict annotations for provided img_ids and cat_ids
        std::vector<std::vector<std::vector<py::dict>>> get_instances(
            const std::vector<double>& img_ids,
            const std::vector<double>& cat_ids, const bool& useCats);

        // Legacy compatibility - same as append_ref but with different
        // signature
        void append(double img_id, double cat_id, const py::dict& ann) {
                append_ref(img_id, cat_id, py::cast<py::object>(ann));
        }

       private:
        // Lightweight storage: only references to Python objects
        std::unordered_map<std::pair<int64_t, int64_t>, std::vector<py::object>,
                           hash_pair>
            annotation_refs;

        // Cache for frequently accessed InstanceAnnotation objects
        mutable std::unordered_map<std::pair<int64_t, int64_t>,
                                   std::vector<InstanceAnnotation>, hash_pair>
            cpp_cache;

        // Helper method to convert py::object to InstanceAnnotation
        InstanceAnnotation parse_py_annotation(const py::object& ann) const;
};
}  // namespace COCOeval
}  // namespace coco_eval
