#include <pybind11/pybind11.h>

#include <iostream>
#include <sstream>

#include "coco_eval/cocoeval.h"
#include "coco_eval/dataset.h"

// Stringification macros for version info
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace coco_eval {

// Returns the compiler version as a string.
// This aids in debugging and reproducibility.
std::string get_compiler_version() {
        std::ostringstream ss;
#if defined(__GNUC__) && !defined(__clang__)
// Ensure GCC version is >= 4.9
#if ((__GNUC__ < 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ < 9)))
#error "GCC >= 4.9 is required!"
#endif
        ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__;
#endif

#if defined(__clang_major__)
        ss << "clang " << __clang_major__ << "." << __clang_minor__ << "."
           << __clang_patchlevel__;
#endif

#if defined(_MSC_VER)
        ss << "MSVC " << _MSC_FULL_VER;
#endif
        return ss.str();
}

PYBIND11_MODULE(faster_eval_api_cpp, m) {
        // Expose utility and COCOeval functions to Python
        m.def("get_compiler_version", &get_compiler_version,
              "Returns the compiler version used for compilation.");
        m.def("COCOevalAccumulate", &COCOeval::Accumulate,
              "Accumulates evaluation statistics.");
        m.def("COCOevalEvaluateImages", &COCOeval::EvaluateImages,
              "Evaluates images based on detections and ground truth.");
        m.def("COCOevalEvaluateAccumulate", &COCOeval::EvaluateAccumulate,
              "Performs evaluation and accumulation in one step.");
        m.def("calc_auc", &COCOeval::calc_auc,
              "Calculates area under curve (AUC) for PR curve.");

        // Expose InstanceAnnotation with its constructor
        pybind11::class_<COCOeval::InstanceAnnotation>(m, "InstanceAnnotation")
            .def(pybind11::init<uint64_t, double, double, bool, bool, bool>());

        // Expose ImageEvaluation with pickle support for serialization
        pybind11::class_<COCOeval::ImageEvaluation>(m, "ImageEvaluation")
            .def(pybind11::init<>())
            .def(pybind11::pickle(
                // __getstate__ for pickling
                [](const COCOeval::ImageEvaluation &p) {
                        // Use reserve to avoid reallocations (performance
                        // optimization)
                        std::vector<std::tuple<uint64_t, uint64_t, double>>
                            matched_annotations;
                        matched_annotations.reserve(
                            p.matched_annotations.size());
                        for (const auto &ann : p.matched_annotations) {
                                matched_annotations.emplace_back(
                                    ann.dt_id, ann.gt_id, ann.iou);
                        }
                        return pybind11::make_tuple(
                            p.detection_matches, p.ground_truth_matches,
                            p.detection_scores, p.ground_truth_ignores,
                            p.detection_ignores, matched_annotations);
                },
                // __setstate__ for unpickling
                [](pybind11::tuple t) {
                        if (t.size() != 6)
                                throw std::runtime_error(
                                    "Invalid state for ImageEvaluation!");
                        COCOeval::ImageEvaluation p;
                        p.detection_matches = t[0].cast<std::vector<int64_t>>();
                        p.ground_truth_matches =
                            t[1].cast<std::vector<int64_t>>();
                        p.detection_scores = t[2].cast<std::vector<double>>();
                        p.ground_truth_ignores = t[3].cast<std::vector<bool>>();
                        p.detection_ignores = t[4].cast<std::vector<bool>>();
                        std::vector<std::tuple<uint64_t, uint64_t, double>>
                            matched_annotations = t[5].cast<std::vector<
                                std::tuple<uint64_t, uint64_t, double>>>();
                        p.matched_annotations.reserve(
                            matched_annotations.size());
                        for (const auto &tup : matched_annotations) {
                                p.matched_annotations.emplace_back(
                                    std::get<0>(tup), std::get<1>(tup),
                                    std::get<2>(tup));
                        }
                        return p;
                }));

        // Expose Dataset with methods and pickle support
        pybind11::class_<COCOeval::Dataset>(m, "Dataset")
            .def(pybind11::init<>())
            .def("append", &COCOeval::Dataset::append)
            .def("clean", &COCOeval::Dataset::clean)
            .def("get", &COCOeval::Dataset::get)
            .def("get_instances", &COCOeval::Dataset::get_instances)
            .def("get_cpp_annotations", &COCOeval::Dataset::get_cpp_annotations)
            .def("get_cpp_instances", &COCOeval::Dataset::get_cpp_instances)
            .def("__len__", [](const COCOeval::Dataset &p) { return p.size(); })
            .def(pybind11::pickle(
                [](const COCOeval::Dataset &p) {
                        // Recommend: Ensure make_tuple() is efficient and
                        // returns minimal necessary state.
                        return p.make_tuple();
                },
                [](pybind11::tuple t) {
                        COCOeval::Dataset p;
                        p.load_tuple(t);
                        return p;
                }));

        // Set the version attribute
#ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
        m.attr("__version__") = "dev";
#endif
}

}  // namespace coco_eval
