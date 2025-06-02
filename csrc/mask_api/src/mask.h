// Copyright (c) MiXaiLL76
#pragma once

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cmath>
#include <vector>

namespace py = pybind11;

typedef unsigned int uint;

namespace mask_api {

namespace Mask {

class RLE {
       public:
        RLE() : h{0}, w{0}, m{0} {}

        RLE(uint64_t h, uint64_t w, uint64_t m, std::vector<uint64_t> cnts)
            : h{h}, w{w}, m{m}, cnts{cnts} {}

        RLE(uint64_t h, uint64_t w, std::vector<uint64_t> cnts)
            : h{h}, w{w}, m{1}, cnts{cnts} {}

        uint64_t h;
        uint64_t w;
        uint64_t m;
        std::vector<uint64_t> cnts;

        std::string toString() const;
        std::tuple<uint64_t, uint64_t, std::string> toTuple() const;
        std::vector<double> toBbox() const;
        RLE erode_3x3(int dilation) const;
        RLE toBoundary(double dilation_ratio) const;
        RLE clear_duplicates() const;
        uint64_t area() const;
        py::dict toDict() const;

        static RLE frString(const std::string &s, uint64_t h, uint64_t w);
        static RLE frBbox(const std::vector<double> &bb, uint64_t h,
                          uint64_t w);
        static RLE frPoly(const std::vector<double> &xy, uint64_t h,
                          uint64_t w);
        static RLE merge(const std::vector<RLE> &R, int intersect);
        static RLE frUncompressedRLE(const py::dict &ucRle);
        static RLE frSegm(const py::object &pyobj, uint64_t w, uint64_t h);
        static RLE frTuple(
            const std::tuple<uint64_t, uint64_t, std::string> &w_h_rlestring);
};

std::vector<py::dict> erode_3x3(const std::vector<py::dict> &rleObjs,
                                const int &dilation = 1);

std::vector<py::dict> toBoundary(const std::vector<py::dict> &rleObjs,
                                 const double &dilation_ratio);

py::array_t<uint8_t, py::array::f_style> rleDecode(const std::vector<RLE> &R);
std::vector<RLE> rleEncode(const py::array_t<uint8_t, py::array::f_style> &M,
                           uint64_t h, uint64_t w, uint64_t n);

py::bytes rleToString(const RLE &R);
RLE rleFrString(const std::string &s, const uint64_t &h, const uint64_t &w);

std::vector<RLE> rleFrBbox(const std::vector<double> &bb, uint64_t h,
                           uint64_t w, uint64_t n);
RLE rleFrPoly(const std::vector<double> &xy, const uint64_t &k,
              const uint64_t &h, const uint64_t &w);

// pyx functions
py::array_t<uint8_t, py::array::f_style> decode(const std::vector<py::dict> &R);
std::vector<py::dict> encode(const py::array_t<uint8_t, py::array::f_style> &M);

py::array_t<double> toBbox(const std::vector<py::dict> &R);
py::dict merge(const std::vector<py::dict> &rleObjs, const int &intersect);
py::dict merge(const std::vector<py::dict> &rleObjs);
py::array_t<uint64_t> area(const std::vector<py::dict> &rleObjs);
std::variant<py::array_t<double, py::array::f_style>, std::vector<double>> iou(
    const py::object &dt, const py::object &gt,
    const std::vector<int> &iscrowd);
std::vector<double> bbIou(const std::vector<double> &dt,
                          const std::vector<double> &gt, std::size_t m,
                          std::size_t n, const std::vector<int> &iscrowd);
std::vector<double> rleIou(const std::vector<RLE> &dt,
                           const std::vector<RLE> &gt, const uint64_t &m,
                           const uint64_t &n, const std::vector<int> &iscrowd);
std::vector<py::dict> _toString(const std::vector<RLE> &R);
std::vector<RLE> _frString(const std::vector<py::dict> &R);
std::vector<py::dict> frPoly(const std::vector<std::vector<double>> &poly,
                             const uint64_t &h, const uint64_t &w);
std::vector<py::dict> frBbox(const std::vector<std::vector<double>> &bb,
                             const uint64_t &h, const uint64_t &w);
std::vector<py::dict> frUncompressedRLE(const std::vector<py::dict> &ucRles);
std::vector<py::dict> toUncompressedRLE(const std::vector<py::dict> &Rles);
std::vector<py::dict> rleToUncompressedRLE(const std::vector<RLE> &R);

py::array rleToBbox(const std::vector<RLE> &R,
                    std::optional<uint64_t> n = std::nullopt);

std::variant<pybind11::dict, std::vector<pybind11::dict>> frPyObjects(
    const py::object &pyobj, const uint64_t &h, const uint64_t &w);
std::variant<pybind11::dict, py::object> segmToRle(const py::object &pyobj,
                                                   const uint64_t &w,
                                                   const uint64_t &h);
std::vector<py::dict> processRleToBoundary(const std::vector<RLE> &rles,
                                           const double &dilation_ratio,
                                           const size_t &cpu_count);
void calculateRleForAllAnnotations(
    const std::vector<py::dict> &anns,
    const std::unordered_map<uint64_t, std::tuple<uint64_t, uint64_t>>
        &image_info,
    const bool &compute_rle, const bool &compute_boundary,
    const double &dilation_ratio, const size_t &cpu_count);
}  // namespace Mask
}  // namespace mask_api
