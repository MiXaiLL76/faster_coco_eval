// Copyright (c) Facebook, Inc. and its affiliates.

#include "coco_eval/cocoeval.h"
#include <iostream>
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace coco_eval
{

  // similar to
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Version.cpp
  std::string get_compiler_version()
  {
    std::ostringstream ss;
#if defined(__GNUC__)
#ifndef __clang__

#if ((__GNUC__ <= 4) && (__GNUC_MINOR__ <= 8))
#error "GCC >= 4.9 is required!"
#endif

    {
      ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__;
    }
#endif
#endif

#if defined(__clang_major__)
    {
      ss << "clang " << __clang_major__ << "." << __clang_minor__ << "."
         << __clang_patchlevel__;
    }
#endif

#if defined(_MSC_VER)
    {
      ss << "MSVC " << _MSC_FULL_VER;
    }
#endif
    return ss.str();
  }

  PYBIND11_MODULE(faster_eval_api_cpp, m)
  {
    m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");
    m.def("COCOevalAccumulate", &COCOeval::Accumulate, "COCOeval::Accumulate");
    m.def("COCOevalEvaluateImages", &COCOeval::EvaluateImages, "COCOeval::EvaluateImages");
    m.def("COCOevalEvaluateAccumulate", &COCOeval::EvaluateAccumulate, "COCOeval::EvaluateAccumulate");

    // slow!
    m.def("_summarize", &COCOeval::_summarize, "COCOeval::_summarize");
    m.def("calc_auc", &COCOeval::calc_auc, "COCOeval::calc_auc");

    pybind11::class_<COCOeval::InstanceAnnotation>(m, "InstanceAnnotation")
    .def(pybind11::init<uint64_t, double, double, bool, bool, bool>());

    pybind11::class_<COCOeval::ImageEvaluation>(m, "ImageEvaluation").def(pybind11::init<>())
    .def(py::pickle(
        [](const COCOeval::ImageEvaluation &p) {

            std::vector<std::tuple<uint64_t, uint64_t, double>> matched_annotations;
            for (size_t i = 0; i < p.matched_annotations.size(); i++) {
                matched_annotations.push_back(std::make_tuple(p.matched_annotations[i].dt_id, p.matched_annotations[i].gt_id, p.matched_annotations[i].iou));
            }

            return py::make_tuple(p.detection_matches, p.ground_truth_matches, p.detection_scores, p.ground_truth_ignores, p.detection_ignores, matched_annotations);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 6)
                throw std::runtime_error("Invalid state!");

            COCOeval::ImageEvaluation p;
            p.detection_matches = t[0].cast<std::vector<int64_t>>();
            p.ground_truth_matches = t[1].cast<std::vector<int64_t>>();
            p.detection_scores = t[2].cast<std::vector<double>>();
            p.ground_truth_ignores = t[3].cast<std::vector<bool>>();
            p.detection_ignores = t[4].cast<std::vector<bool>>();
            std::vector<std::tuple<uint64_t, uint64_t, double>> matched_annotations = t[5].cast<std::vector<std::tuple<uint64_t, uint64_t, double>>>();
            for (size_t i = 0; i < matched_annotations.size(); i++) {
                p.matched_annotations.emplace_back(std::get<0>(matched_annotations[i]), std::get<1>(matched_annotations[i]), std::get<2>(matched_annotations[i]));
            }
            return p;
        }
    ));

    pybind11::class_<COCOeval::Dataset>(m, "Dataset").def(pybind11::init<>())
    .def("append", &COCOeval::Dataset::append)
    .def("clean", &COCOeval::Dataset::clean)
    .def("get", &COCOeval::Dataset::get)
    .def("get_instances", &COCOeval::Dataset::get_instances)
    .def("get_cpp_annotations", &COCOeval::Dataset::get_cpp_annotations)
    .def("get_cpp_instances", &COCOeval::Dataset::get_cpp_instances)
    .def(py::pickle(
        [](const COCOeval::Dataset &p) {
            return p.make_tuple();
        },
        [](py::tuple t) { // __setstate__
            COCOeval::Dataset p;
            p.load_tuple(t);
            return p;
        }
    ));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
  }

} // namespace coco_eval
