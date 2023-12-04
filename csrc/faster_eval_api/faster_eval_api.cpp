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
    pybind11::class_<COCOeval::InstanceAnnotation>(m, "InstanceAnnotation").def(pybind11::init<uint64_t, double, double, bool, bool>());
    pybind11::class_<COCOeval::ImageEvaluation>(m, "ImageEvaluation").def(pybind11::init<>());

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
  }

} // namespace coco_eval