// Copyright (c) MiXaiLL76
//
// Python C++ extension module for efficient mask encoding, decoding, and
// manipulation using pybind11. Provides the RLE (Run-Length Encoding) class and
// a comprehensive set of mask operations for computer vision pipelines.
//
// Features:
//   - RLE encoding/decoding for binary masks, polygons, and bounding boxes
//   - Morphological operations (erosion, boundary extraction)
//   - Conversion between different mask representations (RLE, polygon, bbox,
//   uncompressed)
//   - Calculation of mask area, bounding boxes, and IoU (Intersection over
//   Union) metrics
//   - Batch utilities for encoding and processing mask annotations
//   - Compiler version reporting for diagnostics
//
// Module name: `mask_api_new_cpp`
//
// Example usage (Python):
//   import mask_api_new_cpp as mask
//   rle = mask.RLE(...)               # construct an RLE object
//   area = rle.area()                 # compute mask area
//   bbox = rle.toBbox()               # get bounding box
//   boundary = rle.toBoundary(0.008)  # get boundary mask with given dilation
//   ratio

#include <pybind11/pybind11.h>

#include <iostream>
#include <sstream>

#include "src/mask.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace mask_api {

// Returns a string with the compiler version used to build the module.
// Parameters: none
// Returns:
//   - String indicating the compiler and version (e.g., "GCC 9.3",
//   "clang 12.0.0", "MSVC 1928")
std::string get_compiler_version() {
        std::ostringstream ss;
#if defined(__GNUC__)
#ifndef __clang__
#if ((__GNUC__ <= 4) && (__GNUC_MINOR__ <= 8))
#error "GCC >= 4.9 is required!"
#endif
        ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__;
#endif
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

PYBIND11_MODULE(mask_api_new_cpp, m) {
        // Exposes the RLE (Run-Length Encoding) class for binary masks.
        pybind11::class_<Mask::RLE>(m, "RLE")
            .def(pybind11::init<uint64_t, uint64_t, uint64_t,
                                std::vector<uint64_t>>())
            .def(pybind11::init<>(&Mask::RLE::frString))
            .def(pybind11::init<>(&Mask::RLE::frBbox))
            .def(pybind11::init<>(&Mask::RLE::frPoly))
            .def(pybind11::init<>(&Mask::RLE::merge))
            .def(pybind11::init<>(&Mask::RLE::frUncompressedRLE))
            .def(pybind11::init<>(&Mask::RLE::frSegm))
            .def(pybind11::init<>(&Mask::RLE::frTuple))
            .def("toString", &Mask::RLE::toString,
                 py::call_guard<py::gil_scoped_release>())
            .def("toBbox", &Mask::RLE::toBbox,
                 py::call_guard<py::gil_scoped_release>())
            .def("erode_3x3", &Mask::RLE::erode_3x3,
                 py::call_guard<py::gil_scoped_release>())
            .def("area", &Mask::RLE::area,
                 py::call_guard<py::gil_scoped_release>())
            .def("toBoundary", &Mask::RLE::toBoundary,
                 py::call_guard<py::gil_scoped_release>())
            .def("toDict", &Mask::RLE::toDict)
            .def(py::pickle(
                [](const Mask::RLE &p) {  // __getstate__
                        // Returns a tuple encoding all RLE fields.
                        return py::make_tuple(p.h, p.w, p.m, p.cnts);
                },
                [](py::tuple t) {  // __setstate__
                        if (t.size() != 4)
                                throw std::runtime_error("Invalid state!");
                        // Reconstructs RLE object from the tuple.
                        Mask::RLE p =
                            Mask::RLE(t[0].cast<std::uint64_t>(),
                                      t[1].cast<std::uint64_t>(),
                                      t[2].cast<std::uint64_t>(),
                                      t[3].cast<std::vector<uint64_t>>());
                        return p;
                }));

        // Module-level utility functions for mask encoding, decoding, and
        // manipulation.

        // Returns the compiler version as a string.
        m.def("get_compiler_version", &get_compiler_version,
              "get_compiler_version");

        // Erodes a binary mask using a 3x3 structuring element.
        // Parameters:
        //   - rle: RLE mask to erode
        // Returns:
        //   - RLE mask after erosion
        m.def("erode_3x3", &Mask::erode_3x3, "Mask::erode_3x3");

        // Returns the boundary of the mask as a new RLE, using morphological
        // erosion and XOR. Parameters:
        //   - rle: RLE mask to process
        //   - dilation_ratio: Ratio of mask diagonal for erosion radius
        // Returns:
        //   - RLE mask representing the boundary pixels
        m.def("toBoundary", &Mask::toBoundary, "Mask::toBoundary");

        // Encodes/decodes a binary mask to/from RLE format.
        m.def("rleEncode", &Mask::rleEncode, "Mask::rleEncode");
        m.def("rleDecode", &Mask::rleDecode, "Mask::rleDecode");

        // Converts RLE to/from string representation.
        m.def("rleToString", &Mask::rleToString, "Mask::rleToString");
        m.def("rleFrString", &Mask::rleFrString, "Mask::rleFrString");

        // Converts RLE to/from bounding box representation.
        m.def("rleToBbox", &Mask::rleToBbox, "Mask::rleToBbox");
        m.def("rleFrBbox", &Mask::rleFrBbox, "Mask::rleFrBbox");

        // Converts polygons to RLE.
        m.def("rleFrPoly", &Mask::rleFrPoly, "Mask::rleFrPoly");

        // Internal (pyx) functions for working with RLE and string.
        m.def("_toString", &Mask::_toString, "Mask::_toString");
        m.def("_frString", &Mask::_frString, "Mask::_frString");

        // General encode/decode utilities from/to Python objects.
        m.def("encode", &Mask::encode, "Mask::encode");
        m.def("decode", &Mask::decode, "Mask::decode");

        // Computes intersection-over-union (IoU) between RLE masks or bounding
        // boxes.
        m.def("iou", &Mask::iou, "Mask::iou");
        m.def("bbIou", &Mask::bbIou, "Mask::bbIou");
        m.def("rleIou", &Mask::rleIou, "Mask::rleIou");

        // Converts RLE, polygons, or bounding boxes to the respective
        // representations.
        m.def("toBbox", &Mask::toBbox, "Mask::toBbox");
        m.def("merge",
              py::overload_cast<const std::vector<py::dict> &, const int &>(
                  &Mask::merge),
              "Mask::merge");
        m.def("merge",
              py::overload_cast<const std::vector<py::dict> &>(&Mask::merge),
              "Mask::merge");

        // Computes the area (number of nonzero pixels) of a mask.
        m.def("area", &Mask::area, "Mask::area");

        // Converts polygons and bounding boxes to RLE.
        m.def("frPoly", &Mask::frPoly, "Mask::frPoly");
        m.def("frBbox", &Mask::frBbox, "Mask::frBbox");

        // Converts RLE to/from uncompressed representations.
        m.def("rleToUncompressedRLE", &Mask::rleToUncompressedRLE,
              "Mask::rleToUncompressedRLE");
        m.def("frUncompressedRLE", &Mask::frUncompressedRLE,
              "Mask::frUncompressedRLE");
        m.def("toUncompressedRLE", &Mask::toUncompressedRLE,
              "Mask::toUncompressedRLE");

        // Converts Python segmentation objects to RLE or dict representations.
        m.def("frPyObjects", &Mask::frPyObjects, "Mask::frPyObjects");
        m.def("segmToRle", &Mask::segmToRle, "Mask::segmToRle");

        // Batch utility to compute and attach RLEs and boundaries to annotation
        // objects.
        m.def("calculateRleForAllAnnotations",
              &Mask::calculateRleForAllAnnotations,
              "Mask::calculateRleForAllAnnotations");

        // Sets the module version string.
#ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
        m.attr("__version__") = "dev";
#endif
}

}  // namespace mask_api
