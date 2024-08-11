// Copyright (c) MiXaiLL76

#include "src/mask.h"
#include <iostream>
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace mask_api
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

  PYBIND11_MODULE(mask_api_new_cpp, m)
  {
    pybind11::class_<Mask::RLE>(m, "RLE").def(pybind11::init<uint64_t, uint64_t, uint64_t, std::vector<uint>>());

    m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");

    m.def("rleErode_3x3", &Mask::rleErode_3x3, "Mask::rleErode_3x3");
    m.def("erode_3x3", &Mask::erode_3x3, "Mask::erode_3x3");

    m.def("rleToBoundary", &Mask::rleToBoundary, "Mask::rleToBoundary");
    m.def("toBoundary", &Mask::toBoundary, "Mask::toBoundary");

    m.def("rleEncode", &Mask::rleEncode, "Mask::rleEncode");
    m.def("rleDecode", &Mask::rleDecode, "Mask::rleDecode");

    m.def("rleToString", &Mask::rleToString, "Mask::rleToString");
    m.def("rleFrString", &Mask::rleFrString, "Mask::rleFrString");

    m.def("rleToBbox", &Mask::rleToBbox, "Mask::rleToBbox");
    m.def("rleFrBbox", &Mask::rleFrBbox, "Mask::rleFrBbox");

    m.def("rleMerge", &Mask::rleMerge, "Mask::rleMerge");

    m.def("rleFrPoly", &Mask::rleFrPoly, "Mask::rleFrPoly");

    m.def("rleArea", &Mask::rleArea, "Mask::rleArea");

    // pyx functions
    m.def("_toString", &Mask::_toString, "Mask::_toString");
    m.def("_frString", &Mask::_frString, "Mask::_frString");

    m.def("encode", &Mask::encode, "Mask::encode");
    m.def("decode", &Mask::decode, "Mask::decode");
    m.def("iou", &Mask::iou, "Mask::iou");

    m.def("toBbox", &Mask::toBbox, "Mask::toBbox");
    m.def("merge", py::overload_cast<const std::vector<py::dict> &, const int &>(&Mask::merge), "Mask::merge");
    m.def("merge", py::overload_cast<const std::vector<py::dict> &>(&Mask::merge), "Mask::merge");

    m.def("area", &Mask::area, "Mask::area");
    m.def("bbIou", &Mask::bbIou, "Mask::bbIou");
    m.def("rleIou", &Mask::rleIou, "Mask::rleIou");

    m.def("frPoly", &Mask::frPoly, "Mask::frPoly");
    m.def("frBbox", &Mask::frBbox, "Mask::frBbox");
    m.def("rleToUncompressedRLE", &Mask::rleToUncompressedRLE, "Mask::rleToUncompressedRLE");
    m.def("frUncompressedRLE", &Mask::frUncompressedRLE, "Mask::frUncompressedRLE");
    m.def("toUncompressedRLE", &Mask::toUncompressedRLE, "Mask::toUncompressedRLE");
    m.def("frPyObjects", &Mask::frPyObjects, "Mask::frPyObjects");
    m.def("segmToRle", &Mask::segmToRle, "Mask::segmToRle");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
  }

} // namespace coco_eval