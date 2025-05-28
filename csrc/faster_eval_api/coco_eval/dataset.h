#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace coco_eval {

namespace COCOeval {
template <class T>
inline void hash_combine(std::size_t &seed, const T &v) {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct hash_pair {
        std::size_t operator()(const std::pair<int64_t, int64_t> &p) const {
                std::size_t h = 0;
                hash_combine(h, p.first);
                hash_combine(h, p.second);
                return h;
        }
};

class Dataset {
       public:
        Dataset() {
                // Reserve initial space to reduce rehashing, improving memory
                // and performance.
                data.reserve(8192);
                // Optionally, you can set max_load_factor to lower value if
                // memory is less critical: data.max_load_factor(0.25f);
        }

        // Append a new annotation for (img_id, cat_id) key.
        void append(double img_id, double cat_id, const py::dict &ann);

        // Remove all stored annotations and free memory.
        void clean();

        // Get dataset size (number of (img_id, cat_id) pairs with annotations).
        size_t size() const;

        // Pickle support: Serialize dataset contents to a tuple.
        py::tuple make_tuple() const;

        // Pickle support: Load dataset contents from a tuple.
        void load_tuple(py::tuple pickle_data);

        // Get all Python dict annotations for a given image/category pair.
        std::vector<py::dict> get(double img_id, double cat_id);

        // Get C++ annotation objects for a given image/category pair.
        std::vector<InstanceAnnotation> get_cpp_annotations(double img_id,
                                                            double cat_id);

        // Get all C++ annotation objects for provided img_ids and cat_ids. If
        // useCats is false, cat_ids is ignored.
        std::vector<std::vector<std::vector<InstanceAnnotation>>>
        get_cpp_instances(const std::vector<double> &img_ids,
                          const std::vector<double> &cat_ids,
                          const bool &useCats);

        // Get all Python dict annotations for provided img_ids and cat_ids. If
        // useCats is false, cat_ids is ignored.
        std::vector<std::vector<std::vector<py::dict>>> get_instances(
            const std::vector<double> &img_ids,
            const std::vector<double> &cat_ids, const bool &useCats);

       private:
        // Use unordered_map to store annotations for (img_id, cat_id) pairs.
        // Custom hash functor is used.
        std::unordered_map<std::pair<int64_t, int64_t>, std::vector<py::dict>,
                           hash_pair>
            data;
};
}  // namespace COCOeval
}  // namespace coco_eval
