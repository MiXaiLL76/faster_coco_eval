#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <mutex>
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

        // A std::mutex member is neither copyable nor movable, which would
        // otherwise delete these implicitly and break the pickle __setstate__
        // factory that returns a dataset by value. Move the containers and
        // leave each object with its own fresh mutex; the source is locked so
        // the move cannot race a concurrent reader.
        LightweightDataset(LightweightDataset&& other) noexcept;
        LightweightDataset(const LightweightDataset&) = delete;
        LightweightDataset& operator=(const LightweightDataset&) = delete;

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

        // Get C++ annotation objects with caching for performance.
        //
        // Returns by value rather than by reference: the cached entry it would
        // otherwise expose can be erased by clear_cache_entry from another
        // thread, so a reference outliving the lock would dangle. Every caller
        // copied the result anyway.
        std::vector<InstanceAnnotation> get_cpp_annotations(
            double img_id, double cat_id) const;

        // Clear cache entry for specific (img_id, cat_id) to free memory
        void clear_cache_entry(double img_id, double cat_id) const;

        // Get all C++ annotation objects for provided img_ids and cat_ids
        std::vector<std::vector<std::vector<InstanceAnnotation>>>
        get_cpp_instances(const std::vector<double>& img_ids,
                          const std::vector<double>& cat_ids,
                          const bool& useCats) const;

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
        using AnnotationKey = std::pair<int64_t, int64_t>;
        using AnnotationRefs =
            std::unordered_map<AnnotationKey, std::vector<py::object>,
                               hash_pair>;
        using CppCache =
            std::unordered_map<AnnotationKey, std::vector<InstanceAnnotation>,
                               hash_pair>;

        // Lightweight storage: only references to Python objects
        AnnotationRefs annotation_refs;

        // Cache for frequently accessed InstanceAnnotation objects
        mutable CppCache cpp_cache;

        // Increments whenever annotations are replaced or appended. Parsed
        // data may be published only when this version still matches its
        // source snapshot.
        uint64_t annotation_version = 0;

        // Helper method to convert py::object to InstanceAnnotation
        InstanceAnnotation parse_py_annotation(const py::object& ann) const;

        // Guards annotation_refs, cpp_cache, and annotation_version. Python
        // conversions and Python reference destruction must occur outside it.
        mutable std::mutex mutex;
};
}  // namespace COCOeval
}  // namespace coco_eval
