// Copyright (c) MiXaiLL76
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

namespace py = pybind11;

namespace mask_api
{

    namespace Mask
    {
        struct RLE
        {
            RLE(
                uint64_t h,
                uint64_t w,
                uint64_t m,
                std::vector<uint> cnts) : h{h}, w{w}, m{m}, cnts{cnts} {}

            RLE(
                uint64_t h,
                uint64_t w,
                std::vector<uint> cnts) : h{h}, w{w}, m{1}, cnts{cnts} {}

            uint64_t h;
            uint64_t w;
            uint64_t m;
            std::vector<uint> cnts;
        };

        py::array_t<uint, py::array::f_style> rleDecode(const std::vector<RLE> &R);

        std::vector<RLE> rleEncode(const py::array_t<uint, py::array::f_style> &M, const uint64_t &h, const uint64_t &w, const uint64_t &n);
        py::bytes rleToString(const RLE &R);
        RLE rleFrString(const std::string &s, const uint64_t &h, const uint64_t &w);

        std::vector<RLE> rleFrBbox(const std::vector<double> &bb, const uint64_t &h, const uint64_t &w, const uint64_t &n);
        RLE rleFrPoly(const std::vector<double> &xy, const uint64_t &k, const uint64_t &h, const uint64_t &w);

        std::vector<uint> rleArea(const std::vector<RLE> &R);
        RLE rleMerge(const std::vector<RLE> &R, const int &intersect);

        // pyx functions
        py::array_t<uint, py::array::f_style> decode(const std::vector<py::dict> &R);
        std::vector<py::dict> encode(const py::array_t<uint, py::array::f_style> &M);

        py::array_t<double> toBbox(const std::vector<py::dict> &R);
        py::dict merge(const std::vector<py::dict> &rleObjs, const uint64_t &intersect);
        py::dict merge(const std::vector<py::dict> &rleObjs);
        py::array_t<uint> area(const std::vector<py::dict> &rleObjs);
        std::variant<py::array_t<double, py::array::f_style>, std::vector<double>> iou(const py::object &dt, const py::object &gt, const std::vector<int> &iscrowd);
        std::vector<double> bbIou(const std::vector<double> &dt, const std::vector<double> &gt, const uint64_t &m, const uint64_t &n, const std::vector<int> &iscrowd);
        std::vector<double> rleIou(const std::vector<RLE> &dt, const std::vector<RLE> &gt, const uint64_t &m, const uint64_t &n, const std::vector<int> &iscrowd);
        std::vector<py::dict> _toString(const std::vector<RLE> &R);
        std::vector<RLE> _frString(const std::vector<py::dict> &R);
        std::vector<py::dict> frPoly(const std::vector<std::vector<double>> &poly, const uint64_t &h, const uint64_t &w);
        std::vector<py::dict> frBbox(const std::vector<std::vector<double>> &bb, const uint64_t &h, const uint64_t &w);
        std::vector<py::dict> frUncompressedRLE(const std::vector<py::dict> &ucRles, const uint64_t &h, const uint64_t &w);
        std::vector<py::dict> toUncompressedRLE(const std::vector<py::dict> &Rles);
        std::vector<py::dict> rleToUncompressedRLE(const std::vector<RLE> &R);
        py::array_t<double> rleToBbox(const std::vector<RLE> R, const uint64_t &n);
        std::variant<pybind11::dict, std::vector<pybind11::dict>> frPyObjects(const py::object &pyobj, const uint64_t &h, const uint64_t &w);
        std::variant<pybind11::dict, py::object> segmToRle(const py::object &pyobj, const uint64_t &h, const uint64_t &w);
    } // namespace Mask
} // namespace mask_api