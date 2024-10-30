// Copyright (c) MiXaiLL76
#include "mask.h"
#include <time.h>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <iostream>
#include <execution>
#include <future>
#include <thread>

using namespace pybind11::literals;

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> &vec)
{
    std::vector<T> result;
    for (const auto &v : vec)
        result.insert(result.end(), v.begin(), v.end());
    return result;
}

namespace mask_api
{

    namespace Mask
    {
        uint umin(uint a, uint b) { return (a < b) ? a : b; }
        uint umin(uint8_t a, uint8_t b) { return (a < b) ? a : b; }
        uint umax(uint a, uint b) { return (a > b) ? a : b; }
        uint umax(uint8_t a, uint8_t b) { return (a > b) ? a : b; }

        py::bytes rleToString(const RLE &R)
        {
            return py::bytes(R.toString());
        }

        RLE rleFrString(const std::string &s, const uint64_t &h, const uint64_t &w)
        {
            return RLE::frString(s, h, w);
        }

        std::vector<RLE> rleEncode(const py::array_t<uint8_t, py::array::f_style> &M, const uint64_t &h, const uint64_t &w, const uint64_t &n)
        {
            auto mask = M.unchecked<3>();

            std::vector<RLE> RLES;
            for (uint64_t i = 0; i < n; i++)
            {
                std::vector<uint> cnts = {};
                uint8_t p = 0;
                uint c = 0;
                for (uint64_t row = 0; row < (w); row++)
                {
                    for (uint64_t col = 0; col < (h); col++)
                    {

                        if (mask(col, row, i) != p)
                        {
                            cnts.emplace_back(c);
                            c = 0;
                            p = mask(col, row, i);
                        }
                        c += 1;
                    }
                }

                cnts.emplace_back(c);

                RLES.emplace_back(h, w, cnts.size(), cnts);
            }
            return RLES;
        }

        py::array_t<double> rleToBbox(const std::vector<RLE> R, const uint64_t &n)
        {
            std::vector<double> result;
            for (uint64_t i = 0; i < n; i++)
            {
                std::vector<uint> bbox = R[i].toBbox();
                std::copy(bbox.begin(), bbox.end(), std::back_inserter(result));
            }
            return py::array(result.size(), result.data()).reshape({-1, 4});
        }

        py::array_t<double> toBbox(const std::vector<py::dict> &R)
        {
            std::vector<RLE> rles = _frString(R);
            return rleToBbox(rles, rles.size());
        }

        std::vector<RLE> rleFrBbox(const std::vector<double> &bb, const uint64_t &h, const uint64_t &w, const uint64_t &n)
        {
            std::vector<RLE> result;
            for (uint64_t i = 0; i < n; i++)
            {
                result.emplace_back(RLE::frBbox({bb[i * 4], bb[i * 4 + 1], bb[i * 4 + 2], bb[i * 4 + 3]}, h, w));
            }
            return result;
        }

        RLE rleFrPoly(const std::vector<double> &xy, const uint64_t &k, const uint64_t &h, const uint64_t &w)
        {
            return RLE::frPoly(xy, h, w);
        }

        std::vector<py::dict> _toString(const std::vector<RLE> &rles)
        {
            std::vector<py::dict> result;

            for (uint64_t i = 0; i < rles.size(); i++)
            {
                result.push_back(rles[i].toDict());
            }
            return result;
        }

        // internal conversion from compressed RLE format to Python RLEs object
        std::vector<RLE> _frString(const std::vector<py::dict> &R)
        {
            std::vector<RLE> result;
            for (uint64_t i = 0; i < R.size(); i++)
            {
                std::pair<uint64_t, uint64_t> size = R[i]["size"].cast<std::pair<uint64_t, uint64_t>>();
                std::string counts = R[i]["counts"].cast<std::string>();
                result.emplace_back(RLE::frString(counts, size.first, size.second));
            }
            return result;
        }

        std::vector<py::dict> encode(const py::array_t<uint8_t, py::array::f_style> &M)
        {
            return _toString(rleEncode(M, M.shape(0), M.shape(1), M.shape(2)));
        }

        // Decodes n different RLEs that have the same width and height. Write results to M.
        // Returns whether the decoding succeeds or not.
        py::array_t<uint8_t, py::array::f_style> rleDecode(const std::vector<RLE> &R)
        {
            std::vector<uint> result;
            size_t n = R.size();
            if (n > 0)
            {
                uint64_t h = (R[0].h), w = (R[0].w);

                py::array_t<uint8_t, py::array::f_style> M({(size_t)h, (size_t)w, (size_t)n});
                auto mask = M.mutable_unchecked();
                uint64_t s = h * w * n;

                for (uint64_t i = 0; i < n; i++)
                {
                    uint v = 0;
                    size_t x = 0, y = 0, c = 0;
                    for (uint64_t j = 0; j < R[i].m; j++)
                    {
                        for (uint64_t k = 0; k < R[i].cnts[j]; k++)
                        {
                            c += 1;
                            if (c > s)
                            {
                                std::stringstream ss;
                                ss << "Invalid RLE mask representation; out of range HxW=[0;0]->[" << h - 1 << ";" << w - 1 << "] x=" << x << "; y=" << y;
                                throw std::range_error(ss.str());
                            }

                            mask(y, x, i) = v;

                            y += 1;
                            if (y >= h)
                            {
                                y = 0;
                                x += 1;
                            }
                        }
                        v = !v;
                    }
                }
                return M;
            }
            else
            {
                return {};
            }
        }

        // decode mask from compressed list of RLE string or RLEs object
        py::array_t<uint8_t, py::array::f_style> decode(const std::vector<py::dict> &R)
        {
            return rleDecode(_frString(R));
        }

        std::vector<py::dict> erode_3x3(const std::vector<py::dict> &rleObjs, const int &dilation)
        {
            std::vector<RLE> rles = _frString(rleObjs);
            std::transform(rles.begin(), rles.end(), rles.begin(), [dilation](RLE const &rle)
                           { return rle.erode_3x3(dilation); });
            return _toString(rles);
        }

        std::vector<py::dict> toBoundary(const std::vector<py::dict> &rleObjs, const double &dilation_ratio = 0.02)
        {
            std::vector<RLE> rles = _frString(rleObjs);
            std::transform(rles.begin(), rles.end(), rles.begin(), [&dilation_ratio](RLE const &rle)
                           { return rle.toBoundary(dilation_ratio); });

            return _toString(rles);
        }

        py::dict merge(const std::vector<py::dict> &rleObjs, const int &intersect = 0)
        {
            return _toString({RLE::merge(_frString(rleObjs), intersect)})[0];
        }
        py::dict merge(const std::vector<py::dict> &rleObjs)
        {
            return merge(rleObjs, 0);
        }

        py::array_t<uint> area(const std::vector<py::dict> &rleObjs)
        {
            std::vector<RLE> rles = _frString(rleObjs);
            std::vector<uint> areas(rles.size());
            std::transform(rles.begin(), rles.end(), areas.begin(), [](RLE const &rle)
                           { return rle.area(); });
            return py::array(areas.size(), areas.data());
        }

        std::vector<py::dict> frPoly(const std::vector<std::vector<double>> &poly, const uint64_t &h, const uint64_t &w)
        {
            std::vector<RLE> rles;
            for (uint64_t i = 0; i < poly.size(); i++)
            {
                rles.emplace_back(RLE::frPoly(poly[i], h, w));
            }
            return _toString(rles);
        }

        std::vector<py::dict> frBbox(const std::vector<std::vector<double>> &bb, const uint64_t &h, const uint64_t &w)
        {
            std::vector<RLE> rles;
            for (uint64_t i = 0; i < bb.size(); i++)
            {
                rles.emplace_back(RLE::frBbox(bb[i], h, w));
            }
            return _toString(rles);
        }

        std::vector<py::dict> rleToUncompressedRLE(const std::vector<RLE> &R)
        {
            std::vector<py::dict> result;
            for (uint64_t i = 0; i < R.size(); i++)
            {
                std::vector<uint64_t> size = {R[i].h, R[i].w};
                result.push_back(py::dict("size"_a = size, "counts"_a = R[i].cnts));
            }
            return result;
        }

        std::vector<py::dict> toUncompressedRLE(const std::vector<py::dict> &Rles)
        {
            return rleToUncompressedRLE(_frString(Rles));
        }

        std::vector<py::dict> frUncompressedRLE(const std::vector<py::dict> &ucRles)
        {
            std::vector<RLE> rles;
            for (uint64_t i = 0; i < ucRles.size(); i++)
            {
                std::pair<uint64_t, uint64_t> size = ucRles[i]["size"].cast<std::pair<uint64_t, uint64_t>>();
                std::vector<uint> counts = ucRles[i]["counts"].cast<std::vector<uint>>();
                rles.emplace_back(size.first, size.second, counts.size(), counts);
            }
            return _toString(rles);
        }

        std::vector<double> bbIou(const std::vector<double> &dt, const std::vector<double> &gt, const uint64_t &m, const uint64_t &n, const std::vector<int> &iscrowd)
        {
            double h, w, i, u, ga, da;
            uint64_t g, d;
            bool crowd;
            bool _iscrowd = iscrowd.size() > 0;

            std::vector<double> o(m * n, 0);

            for (g = 0; g < n; g++)
            {
                uint64_t offset = g * 4;
                std::vector<double> G(gt.begin() + offset, gt.begin() + offset + 4);
                ga = G[2] * G[3];
                crowd = _iscrowd && iscrowd[g];
                for (d = 0; d < m; d++)
                {
                    uint64_t offset_d = d * 4;
                    std::vector<double> D(dt.begin() + offset_d, dt.begin() + offset_d + 4);
                    da = D[2] * D[3];
                    o[d * n + g] = 0;
                    w = fmin(D[2] + D[0], G[2] + G[0]) - fmax(D[0], G[0]);
                    if (w <= 0)
                        continue;
                    h = fmin(D[3] + D[1], G[3] + G[1]) - fmax(D[1], G[1]);
                    if (h <= 0)
                        continue;
                    i = w * h;
                    u = crowd ? da : da + ga - i;
                    o[d * n + g] = i / u;
                }
            }
            return o;
        }

        std::vector<double> rleIou(const std::vector<RLE> &dt, const std::vector<RLE> &gt, const uint64_t &m, const uint64_t &n, const std::vector<int> &iscrowd)
        {
            uint64_t g, d;
            std::vector<double> db, gb;
            int crowd;

            for (uint64_t i = 0; i < m; i++)
            {
                std::vector<uint> bbox = dt[i].toBbox();
                std::copy(bbox.begin(), bbox.end(), std::back_inserter(db));
            }
            for (uint64_t i = 0; i < n; i++)
            {
                std::vector<uint> bbox = gt[i].toBbox();
                std::copy(bbox.begin(), bbox.end(), std::back_inserter(gb));
            }

            std::vector<double> o = bbIou(db, gb, m, n, iscrowd);
            bool _iscrowd = iscrowd.size() > 0;

            for (g = 0; g < n; g++)
            {
                for (d = 0; d < m; d++)
                {
                    if (o[d * n + g] > 0)
                    {
                        crowd = _iscrowd && iscrowd[g];
                        if (dt[d].h != gt[g].h || dt[d].w != gt[g].w)
                        {
                            o[g * n + d] = -1;
                            continue;
                        }
                        uint64_t ka, kb, a, b;
                        uint c, ca, cb, ct, i, u;
                        int va, vb;
                        ca = dt[d].cnts[0];
                        ka = dt[d].m;
                        va = vb = 0;
                        cb = gt[g].cnts[0];
                        kb = gt[g].m;
                        a = b = 1;
                        i = u = 0;
                        ct = 1;
                        while (ct > 0)
                        {
                            c = umin(ca, cb);
                            if (va || vb)
                            {
                                u += c;
                                if (va && vb)
                                    i += c;
                            }
                            ct = 0;
                            ca -= c;
                            if (!ca && a < ka)
                            {
                                ca = dt[d].cnts[a++];
                                va = !va;
                            }
                            ct += ca;
                            cb -= c;
                            if (!cb && b < kb)
                            {
                                cb = gt[g].cnts[b++];
                                vb = !vb;
                            }
                            ct += cb;
                        }
                        if (i == 0)
                            u = 1;
                        else if (crowd)
                        {
                            u = dt[d].area();
                        }
                        o[d * n + g] = (double)i / (double)u;
                    }
                }
            }
            return o;
        }

        std::vector<double> _preproc_bbox_array(const py::object &pyobj)
        {
            std::vector<std::vector<double>> array = pyobj.cast<std::vector<std::vector<double>>>();
            if ((array.size() > 0) && (array[0].size() == 4))
            {
                return flatten(array);
            }
            else
            {
                throw std::out_of_range("numpy ndarray input is only for *bounding boxes* and should have Nx4 dimension");
            }
        }

        std::tuple<std::variant<std::vector<RLE>, std::vector<double>>, size_t> _preproc(const py::object &pyobj)
        {
            std::string type = py::str(py::type::of(pyobj));
            if (type == "<class 'numpy.ndarray'>")
            {
                std::vector<double> result = _preproc_bbox_array(pyobj);

                return std::make_tuple(result, (size_t)(result.size() / 4));
            }
            else if (type == "<class 'list'>")
            {
                std::vector<py::object> pyobj_list = pyobj.cast<std::vector<py::object>>();

                if (pyobj_list.size() == 0)
                {
                    return std::make_tuple(std::vector<double>(0), 0);
                }

                bool isbox = true;
                bool isrle = true;
                std::string sub_type = py::str(py::type::of(pyobj_list[0]));

                if (sub_type == "<class 'list'>" || sub_type == "<class 'numpy.ndarray'>")
                {
                    std::vector<std::vector<double>> matrix = pyobj.cast<std::vector<std::vector<double>>>();
                    for (size_t i = 0; i < matrix.size(); i++)
                    {
                        if (matrix[i].size() != 4)
                        {
                            isbox = false;
                            break;
                        }
                    }
                }
                else if (sub_type != "<class 'dict'>")
                {
                    isrle = false;
                }
                else if (sub_type == "<class 'dict'>")
                {
                    isbox = false;
                }
                else
                {
                    isbox = false;
                    isrle = false;
                }

                if (isbox)
                {
                    std::vector<double> result = _preproc_bbox_array(pyobj);
                    return std::make_tuple(result, (size_t)(result.size() / 4));
                }
                else if (isrle)
                {
                    std::vector<RLE> result = _frString(pyobj.cast<std::vector<py::dict>>());
                    return std::make_tuple(result, (size_t)result.size());
                }
                else
                {
                    throw std::out_of_range("list input can be bounding box (Nx4) or RLEs ([RLE])");
                }
            }
            else
            {
                throw std::out_of_range("unrecognized type.  The following type: RLEs (rle), np.ndarray (box), and list (box) are supported.");
            }
        }

        // iou computation. support function overload (RLEs-RLEs and bbox-bbox).
        std::variant<py::array_t<double, py::array::f_style>, std::vector<double>> iou(const py::object &dt, const py::object &gt, const std::vector<int> &iscrowd)
        {

            auto [_dt, m] = _preproc(dt);
            auto [_gt, n] = _preproc(gt);

            if (m == 0 || n == 0)
            {
                return std::vector<double>(0);
            }

            if (_dt.index() != _gt.index())
            {
                throw std::out_of_range("The dt and gt should have the same data type, either RLEs, list or np.ndarray");
            }

            std::size_t crowd_length = iscrowd.size();
            std::vector<double> iou;

            if (std::holds_alternative<std::vector<double>>(_dt))
            {
                std::vector<double> _gt_box = std::get<std::vector<double>>(_gt);
                if (crowd_length > 0 && crowd_length == n)
                {

                    std::vector<double> _dt_box = std::get<std::vector<double>>(_dt);
                    m = (std::size_t)(_dt_box.size() / 4);
                    iou = bbIou(_dt_box, _gt_box, m, n, iscrowd);
                }
            }
            else
            {
                std::vector<RLE> _gt_rle = std::get<std::vector<RLE>>(_gt);
                if (crowd_length > 0 && crowd_length == n)
                {
                    std::vector<RLE> _dt_rle = std::get<std::vector<RLE>>(_dt);
                    m = _dt_rle.size();
                    iou = rleIou(_dt_rle, _gt_rle, m, n, iscrowd);
                }
            }

            if (crowd_length > 0 && crowd_length != n)
            {
                printf("crowd_length=%zu, n=%zu\n", crowd_length, n);
                throw std::out_of_range("iscrowd must have the same length as gt");
            }
            return py::array(iou.size(), iou.data()).reshape({m, n});
        }

        std::variant<pybind11::dict, std::vector<pybind11::dict>> frPyObjects(const py::object &pyobj, const uint64_t &h, const uint64_t &w)
        {
            std::vector<RLE> rles;
            std::string type = py::str(py::type::of(pyobj));

            // encode rle from a list of python objects
            if (type == "<class 'list'>")
            {
                std::vector<py::object> pyobj_list = pyobj.cast<std::vector<py::object>>();
                if (pyobj_list.size() == 0)
                {
                    throw std::out_of_range("list index out of range");
                }

                std::string sub_type = py::str(py::type::of(pyobj_list[0]));

                if (sub_type == "<class 'dict'>")
                {
                    return frUncompressedRLE(pyobj.cast<std::vector<py::dict>>());
                }
                else if ((sub_type == "<class 'list'>") || (sub_type == "<class 'numpy.ndarray'>"))
                {
                    std::vector<std::vector<double>> numpy_array = pyobj.cast<std::vector<std::vector<double>>>();
                    if (numpy_array[0].size() == 4)
                    {
                        return frBbox(numpy_array, h, w);
                    }
                    else if (numpy_array[0].size() > 4)
                    {
                        return frPoly(numpy_array, h, w);
                    }
                }
                else if ((sub_type == "<class 'float'>") || (sub_type == "<class 'int'>"))
                {
                    std::vector<double> array = pyobj.cast<std::vector<double>>();
                    if (array.size() == 4)
                    {
                        return frBbox({array}, h, w)[0]; // need return first
                    }
                    else if (array.size() > 4)
                    {
                        return frPoly({array}, h, w)[0]; // need return first
                    }
                }
            }
            // # encode rle from single python object
            else if (type == "<class 'numpy.ndarray'>")
            {
                return frBbox(pyobj.cast<std::vector<std::vector<double>>>(), h, w);
            }
            else if (type == "<class 'dict'>")
            {
                return frUncompressedRLE({pyobj})[0]; // need return first
            }
            else
            {
                throw py::type_error("input type is not supported.");
            }

            return _toString(rles);
        }

        std::variant<pybind11::dict, py::object> segmToRle(const py::object &pyobj, const uint64_t &w, const uint64_t &h)
        {
            try
            {
                RLE rle = RLE::frSegm(pyobj, w, h);
                return rle.toDict();
            }
            catch (py::type_error const &)
            {
                return pyobj;
            }
        }

        std::vector<py::dict> processRleToBoundary(const std::vector<RLE> &rles, const double &dilation_ratio, const size_t &cpu_count)
        {
            py::gil_scoped_release release;
            std::vector<std::tuple<uint64_t, uint64_t, std::string>> result(rles.size());

// Windows not support async
#ifndef _WIN32
            auto process = [&rles, &result](size_t s, size_t e, double d) mutable
            {
                for (size_t i = s; i < e; ++i)
                {
                    result[i] = rles[i].toBoundary(d).toTuple();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            };
#endif

            size_t start = 0;
            size_t step = 1000;
            size_t end = step;

            if (end > rles.size())
            {
                end = rles.size();
            }

            while (start < rles.size())
            {
#ifndef _WIN32
                std::vector<std::future<void>> rle_futures(cpu_count);
#endif

                size_t thread = 0;
                for (thread = 0; thread < cpu_count; thread++)
                {
#ifdef _WIN32
                    for (size_t i = start; i < end; ++i)
                    {
                        result[i] = rles[i].toBoundary(dilation_ratio).toTuple();
                    }
#else
                    rle_futures[thread] = std::async(std::launch::async, process, start, end, dilation_ratio);
#endif

                    start += step;
                    end += step;

                    if (end > rles.size())
                    {
                        end = rles.size();
                    }
                    if (start >= rles.size())
                    {
                        thread++;
                        break;
                    }
                }

#ifndef _WIN32
                for (size_t i = 0; i < thread; i++)
                {
                    rle_futures[i].wait();
                }
                rle_futures.clear();
                rle_futures.shrink_to_fit();
#endif
            }

            py::gil_scoped_acquire acquire;

            std::vector<py::dict> py_result(result.size());

            for (size_t i = 0; i < result.size(); i++)
            {
                py_result[i] = py::dict(
                    "size"_a = std::vector<uint64_t>{std::get<0>(result[i]), std::get<1>(result[i])},
                    "counts"_a = py::bytes(std::get<2>(result[i])));
            }
            return py_result;
        }

        void calculateRleForAllAnnotations(
            const std::vector<py::dict> &anns,
            const std::unordered_map<uint64_t, std::tuple<uint64_t, uint64_t>> &image_info,
            const bool &compute_rle,
            const bool &compute_boundary,
            const double &dilation_ratio,
            const size_t &cpu_count)
        {
            if (compute_rle)
            {
                size_t ann_count = anns.size();
                std::vector<RLE> rles(ann_count);
                for (size_t i = 0; i < ann_count; i++)
                {
                    if (anns[i].contains("segmentation"))
                    {
                        uint64_t image_id = anns[i]["image_id"].cast<uint64_t>();
                        std::tuple<uint64_t, uint64_t> image_hw = image_info.at(image_id);
                        rles[i] = RLE::frSegm(anns[i]["segmentation"], std::get<1>(image_hw), std::get<0>(image_hw));
                    }
                }
                std::vector<py::dict> boundary_array;

                if (compute_boundary)
                {

                    boundary_array = processRleToBoundary(rles, dilation_ratio, cpu_count);
                }

                for (size_t i = 0; i < ann_count; i++)
                {
                    anns[i]["rle"] = rles[i].toDict();
                    if (compute_boundary)
                    {
                        anns[i]["boundary"] = boundary_array[i];
                    }
                }
            }
        }

    } // namespace Mask

}
