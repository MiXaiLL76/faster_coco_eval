// Copyright (c) MiXaiLL76
#include "mask.h"
#include <time.h>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <iostream>
#include <execution>

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

        std::vector<RLE> rleEncode(const py::array_t<uint8_t, py::array::f_style> &M, const uint64_t &h, const uint64_t &w, const uint64_t &n, int padding = 0)
        {
            auto mask = M.unchecked<3>();

            std::vector<RLE> RLES;
            for (uint64_t i = 0; i < n; i++)
            {
                std::vector<uint> cnts;
                uint p = 0;
                uint c = 0;
                for (uint64_t row = padding; row < (w - padding); row++)
                {
                    for (uint64_t col = padding; col < (h - padding); col++)
                    {

                        if (mask(col, row, i) != p)
                        {
                            cnts.push_back(c);
                            c = 0;
                            p = mask(col, row, i);
                        }
                        c += 1;
                    }
                }

                cnts.push_back(c);

                RLES.push_back(RLE(h, w, cnts.size(), cnts));
            }
            return RLES;
        }

        py::bytes rleToString(const RLE &R)
        {
            int64_t x;
            int64_t c;
            bool more;
            std::string result;

            for (uint64_t i = 0; i < R.m; i++)
            {
                x = (int64_t)R.cnts[i];
                if (i > 2)
                {
                    x -= (int64_t)R.cnts[i - 2];
                }

                more = true;

                while (more)
                {
                    c = x & 0x1f;
                    x >>= 5;
                    more = (c & 0x10) ? x != -1 : x != 0;
                    if (more)
                        c |= 0x20;
                    c += 48;
                    result += (char)c;
                }
            }
            return py::bytes(result);
        }

        RLE rleFrString(const std::string &s, const uint64_t &h, const uint64_t &w)
        {
            int64_t x;
            int64_t c;
            int64_t k;
            bool more;
            std::vector<uint> cnts;

            size_t m = s.size();
            size_t i = 0;

            while (i < m)
            {
                x = 0;
                k = 0;
                more = true;

                while (more)
                {
                    c = s[i] - 48;
                    x |= (c & 0x1f) << 5 * k;
                    more = c & 0x20;
                    c &= 0x10 ? -1 : 0;
                    i += 1;
                    k += 1;

                    if (!more && (c & 0x10))
                    {
                        x |= -1 << 5 * k;
                    }
                }

                if (cnts.size() > 2)
                {
                    x += cnts[cnts.size() - 2];
                }

                cnts.emplace_back(x);
            }

            return RLE(h, w, cnts.size(), cnts);
        }

        py::array_t<double> rleToBbox(const std::vector<RLE> R, const uint64_t &n)
        {
            std::vector<double> result;
            for (uint64_t i = 0; i < n; i++)
            {
                uint h, w, xs, ys, xe, ye, cc;
                uint64_t j, m;

                h = (uint)R[i].h;
                w = (uint)R[i].w;
                m = R[i].m;

                m = ((uint64_t)(m / 2)) * 2;

                xs = w;
                ys = h;
                xe = ye = 0;
                cc = 0;
                if (m == 0)
                {
                    result.insert(result.end(), {0, 0, 0, 0});
                    continue;
                }
                for (j = 0; j < m; j++)
                {
                    uint start = cc;    // start of current segment
                    cc += R[i].cnts[j]; // start of next segment
                    if (j % 2 == 0)
                        continue; // skip background segment
                    if (R[i].cnts[j] == 0)
                        continue; // skip zero-length foreground segment
                    uint y_start = start % h, x_start = (start - y_start) / h;
                    uint y_end = (cc - 1) % h, x_end = (cc - 1 - y_end) / h;

                    // x_start <= x_end must be true
                    xs = umin(xs, x_start);
                    xe = umax(xe, x_end);

                    if (x_start < x_end)
                    {
                        ys = 0;
                        ye = h - 1; // foreground segment goes across columns
                    }
                    else
                    {
                        // if x_start == x_end, then y_start <= y_end must be true
                        ys = umin(ys, y_start);
                        ye = umax(ye, y_end);
                    }
                }
                result.insert(result.end(), {(double)xs, (double)ys, (double)(xe - xs + 1), (double)(ye - ys + 1)});
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
                double xs = bb[4 * i + 0], xe = xs + bb[4 * i + 2];
                double ys = bb[4 * i + 1], ye = ys + bb[4 * i + 3];
                std::vector<double> xy = {xs, ys, xs, ye, xe, ye, xe, ys};
                result.push_back(rleFrPoly(xy, 4, h, w));
            }
            return result;
        }

        RLE rleFrPoly(const std::vector<double> &xy, const uint64_t &k, const uint64_t &h, const uint64_t &w)
        {
            /* upsample and get discrete points densely along entire boundary */
            uint64_t j = 0;
            double scale = 5;

            std::vector<int> x(k + 1);
            std::vector<int> y(k + 1);
            for (j = 0; j < k; j++)
            {
                x[j] = (int)(scale * xy[j * 2 + 0] + .5);
                y[j] = (int)(scale * xy[j * 2 + 1] + .5);
            }
            x[k] = x[0];
            y[k] = y[0];

            std::vector<int> u;
            std::vector<int> v;

            for (j = 0; j < k; j++)
            {
                int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1], dx, dy, t, d;
                int flip;
                double s;
                dx = abs(xe - xs);
                dy = abs(ys - ye);
                flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
                if (flip)
                {
                    t = xs;
                    xs = xe;
                    xe = t;
                    t = ys;
                    ys = ye;
                    ye = t;
                }
                s = dx >= dy ? (double)(ye - ys) / dx : (double)(xe - xs) / dy;

                if (dx >= dy)
                    for (d = 0; d <= dx; d++)
                    {
                        t = flip ? dx - d : d;

                        u.emplace_back(t + xs);
                        v.emplace_back((int)(ys + s * t + .5));
                    }
                else
                    for (d = 0; d <= dy; d++)
                    {
                        t = flip ? dy - d : d;

                        v.emplace_back(t + ys);
                        u.emplace_back((int)(xs + s * t + .5));
                    }
            }

            /* get points along y-boundary and downsample */

            double xd, yd;
            x.clear();
            y.clear();

            for (j = 1; j < u.size(); j++)
                if (u[j] != u[j - 1])
                {
                    xd = (double)(u[j] < u[j - 1] ? u[j] : u[j] - 1);
                    xd = (xd + .5) / scale - .5;
                    if (floor(xd) != xd || xd < 0 || xd > w - 1)
                        continue;
                    yd = (double)(v[j] < v[j - 1] ? v[j] : v[j - 1]);
                    yd = (yd + .5) / scale - .5;
                    if (yd < 0)
                        yd = 0;
                    else if (yd > h)
                        yd = h;

                    yd = ceil(yd);
                    x.emplace_back((int)xd);
                    y.emplace_back((int)yd);
                }

            /* compute rle encoding given y-boundary points */
            std::vector<uint> a;
            std::vector<uint> b;

            for (j = 0; j < x.size(); j++)
                a.emplace_back((uint)(x[j] * (int)(h) + y[j]));
            a.emplace_back((uint)(h * w));

            std::stable_sort(a.begin(), a.end());

            uint p = 0;
            for (j = 0; j < a.size(); j++)
            {
                uint t = a[j];
                a[j] -= p;
                p = t;
            }

            j = 1;
            b.emplace_back(a[0]);

            while (j < a.size())
                if (a[j] > 0)
                    b.emplace_back(a[j++]);
                else
                {
                    j++;
                    if (j < a.size())
                        b[b.size() - 1] += a[j++];
                }

            return RLE(h, w, b.size(), b);
        }

        std::vector<py::dict> _toString(const std::vector<RLE> &rles)
        {
            std::vector<py::dict> result;

            for (uint64_t i = 0; i < rles.size(); i++)
            {
                py::bytes c_string = rleToString(rles[i]);
                std::vector<uint64_t> size = {rles[i].h, rles[i].w};
                result.push_back(py::dict("size"_a = size, "counts"_a = c_string));
            }
            return result;
        }

        // internal conversion from compressed RLE format to Python RLEs object
        std::vector<RLE> _frString(const std::vector<py::dict> &R)
        {
            std::vector<RLE> result;
            for (uint64_t i = 0; i < R.size(); i++)
            {
                std::vector<uint64_t> size;
                std::string counts;

                for (auto it : R[i])
                {
                    std::string name = it.first.cast<std::string>();

                    if (name == "size")
                    {
                        size = it.second.cast<std::vector<uint64_t>>();
                    }
                    else if (name == "counts")
                    {
                        counts = it.second.cast<std::string>();
                    }
                }

                if (size.size() == 2)
                {
                    result.push_back(rleFrString(counts, size[0], size[1]));
                }
            }
            return result;
        }

        std::vector<py::dict> encode(const py::array_t<uint8_t, py::array::f_style> &M)
        {
            return _toString(rleEncode(M, M.shape(0), M.shape(1), M.shape(2)));
        }

        // Decodes n different RLEs that have the same width and height. Write results to M.
        // Returns whether the decoding succeeds or not.
        py::array_t<uint8_t, py::array::f_style> rleDecode(const std::vector<RLE> &R, int padding = 0)
        {
            std::vector<uint> result;
            size_t n = R.size();
            if (n > 0)
            {
                uint64_t h = (R[0].h + padding + padding), w = (R[0].w + padding + padding);
                py::array_t<uint8_t, py::array::f_style> M({h, w, n});
                auto mask = M.mutable_unchecked();
                uint64_t s = h * w * n;

                for (uint64_t i = 0; i < n; i++)
                {
                    uint v = 0;
                    size_t x = padding, y = padding, c = 0;
                    for (uint64_t j = 0; j < R[i].m; j++)
                    {
                        for (uint64_t k = 0; k < R[i].cnts[j]; k++)
                        {
                            mask(y, x, i) = v;
                            c += 1;
                            if (c > s)
                            {
                                throw std::range_error("Invalid RLE mask representation");
                            }
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

        RLE rleErode_3x3(const RLE &rle)
        {
            bool v = false;
            int64_t max_len = rle.w * rle.h;
            std::vector<bool> _counts(max_len, false);
            std::vector<bool>::iterator ptr = _counts.begin();
            std::for_each(rle.cnts.begin(), rle.cnts.end(), [&v, &ptr](uint count)
                          {
                
                if(v){
                    std::fill_n(ptr, count, v);
                }
                
                v = !v;
                ptr += count; });

            std::vector<int> ofsvec = {(int)rle.h, 1, (int)rle.h - 1, (int)rle.h + 1};
            std::vector<int> ofsvec_bottom = {1, (int)-rle.h + 1, (int)rle.h + 1};

            int64_t c = 0;
            int64_t ic = 0;
            RLE result(rle.h, rle.w, {});

            bool _min = false, _prev_min = false;

            v = true;
            for (uint j : rle.cnts)
            {
                result.cnts.emplace_back(0);

                v = !v;
                if (v)
                {
                    _prev_min = false;

                    for (uint k = 0; k < j; k++)
                    {
                        if (_prev_min)
                        {
                            _min = std::all_of(ofsvec_bottom.begin(), ofsvec_bottom.end(), [c, max_len, &_counts](int o)
                                               {
                                int64_t test_ptr = c + o;
                                return (
                                    (test_ptr > 0) && 
                                    (test_ptr < max_len) && _counts[test_ptr]
                                ); 
                            });
                        }
                        else
                        {
                            _min = std::all_of(ofsvec.begin(), ofsvec.end(), [c, max_len, &_counts](int o)
                                               {
                                int64_t test_ptr = c + o;
                                int64_t _test_ptr = c - o;
                                return (
                                    (_test_ptr > 0) && _counts[_test_ptr] && 
                                    (test_ptr < max_len) && _counts[test_ptr]
                                ); });
                        }   

                        if (_min)
                        {
                            result.cnts[ic] += 1;
                        }
                        else
                        {
                            if (_prev_min)
                            {
                                result.cnts.insert(result.cnts.end(), {1, 0});
                                ic += 2;
                            }
                            else
                            {
                                result.cnts[ic - 1] += 1;
                            }
                        }
                        _prev_min = _min;

                        c++;
                    }
                }
                else
                {
                    result.cnts[ic] += j;
                    c += j;
                }

                ic++;
            }

            result.m = result.cnts.size();

            return result;
        }

        std::vector<py::dict> erode_3x3(const std::vector<py::dict> &rleObjs)
        {
            std::vector<RLE> rles = _frString(rleObjs);
            std::transform(rles.begin(), rles.end(), rles.begin(), [](RLE const &rle)
                           { return rleErode_3x3(rle); });
            return _toString(rles);
        }

        RLE rleToBoundary(const RLE &rle, const double &dilation_ratio = 0.02)
        {
            int dilation = (int)std::round((dilation_ratio * std::sqrt(rle.h * rle.h + rle.w * rle.w)));
            if (dilation < 1)
            {
                dilation = 1;
            }

            RLE erode_rle(rle);
            for (int i = 0; i < dilation; i++)
            {
                erode_rle = rleErode_3x3(erode_rle);
            }
            return rleMerge({rle, erode_rle}, -1);
        }
        std::vector<py::dict> toBoundary(const std::vector<py::dict> &rleObjs, const double &dilation_ratio)
        {
            std::vector<RLE> rles = _frString(rleObjs);
            std::transform(rles.begin(), rles.end(), rles.begin(), [&dilation_ratio](RLE const &rle)
                           { return rleToBoundary(rle, dilation_ratio); });

            return _toString(rles);
        }

        RLE rleMerge(const std::vector<RLE> &R, const int &intersect)
        {
            uint64_t n = R.size();

            if (n == 0)
            {
                return RLE(0, 0, 0, {});
            }
            else if (n == 1)
            {
                return R[0];
            }
            else
            {
                uint64_t h = R[0].h, w = R[0].w;
                uint c, ca, cb, cc, ct;
                bool v, va, vb, vp;
                uint64_t m = R[0].m;
                uint64_t i, a, b;

                std::vector<uint> cnts(h * w + 1);
                for (a = 0; a < m; a++)
                {
                    cnts[a] = R[0].cnts[a];
                }
                for (i = 1; i < n; i++)
                {
                    RLE B = R[i];
                    if (B.h != h || B.w != w)
                    {
                        h = w = m = 0;
                        break;
                    }
                    RLE A = RLE(h, w, m, cnts);
                    ca = A.cnts[0];
                    cb = B.cnts[0];
                    v = va = vb = false;
                    m = 0;
                    a = b = 1;
                    cc = 0;
                    ct = 1;
                    while (ct > 0)
                    {
                        c = umin(ca, cb);
                        cc += c;
                        ct = 0;
                        ca -= c;
                        if (!ca && a < A.m)
                        {
                            ca = A.cnts[a++];
                            va = !va;
                        }
                        ct += ca;
                        cb -= c;
                        if (!cb && b < B.m)
                        {
                            cb = B.cnts[b++];
                            vb = !vb;
                        }
                        ct += cb;
                        vp = v;

                        if (intersect == 1)
                        {
                            v = va && vb;
                        }
                        else if (intersect == 0)
                        {
                            v = va || vb;
                        }
                        else
                        {
                            v = va != vb;
                        }

                        if (v != vp || ct == 0)
                        {
                            cnts[m++] = cc;
                            cc = 0;
                        }
                    }
                }

                return RLE(h, w, m, cnts);
            }
        }

        py::dict merge(const std::vector<py::dict> &rleObjs, const int &intersect = 0)
        {
            return _toString({rleMerge(_frString(rleObjs), intersect)})[0];
        }
        py::dict merge(const std::vector<py::dict> &rleObjs)
        {
            return merge(rleObjs, 0);
        }

        std::vector<uint> rleArea(const std::vector<RLE> &R)
        {
            uint64_t i, j;
            std::vector<uint> result(R.size(), 0);
            for (i = 0; i < R.size(); i++)
            {
                for (j = 1; j < R[i].m; j += 2)
                    result[i] += R[i].cnts[j];
            }
            return result;
        }
        py::array_t<uint> area(const std::vector<py::dict> &rleObjs)
        {
            std::vector<uint> areas = rleArea(_frString(rleObjs));
            return py::array(areas.size(), areas.data());
        }

        std::vector<py::dict> frPoly(const std::vector<std::vector<double>> &poly, const uint64_t &h, const uint64_t &w)
        {
            std::vector<RLE> rles;
            for (uint64_t i = 0; i < poly.size(); i++)
            {
                rles.push_back(rleFrPoly(poly[i], poly[i].size() / 2, h, w));
            }
            return _toString(rles);
        }

        std::vector<py::dict> frBbox(const std::vector<std::vector<double>> &bb, const uint64_t &h, const uint64_t &w)
        {
            std::vector<RLE> rles;
            for (uint64_t i = 0; i < bb.size(); i++)
            {
                rles.push_back(rleFrBbox(bb[i], h, w, 1)[0]);
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

        std::vector<py::dict> frUncompressedRLE(const std::vector<py::dict> &ucRles, const uint64_t &h, const uint64_t &w)
        {
            std::vector<RLE> rles;
            for (uint64_t i = 0; i < ucRles.size(); i++)
            {
                std::vector<uint64_t> size;
                std::vector<uint> counts;

                for (auto it : ucRles[i])
                {
                    std::string name = it.first.cast<std::string>();

                    if (name == "size")
                    {
                        size = it.second.cast<std::vector<uint64_t>>();
                    }
                    else if (name == "counts")
                    {
                        counts = it.second.cast<std::vector<uint>>();
                    }
                }

                if (size.size() == 2)
                {
                    rles.push_back(RLE(size[0], size[1], counts.size(), counts));
                }
            }
            return _toString(rles);
        }

        std::vector<double> bbIou(const std::vector<double> &dt, const std::vector<double> &gt, const uint64_t &m, const uint64_t &n, const std::vector<int> &iscrowd)
        {
            double h, w, i, u, ga, da;
            uint64_t g, d;
            bool crowd;
            bool _iscrowd = iscrowd.size() > 0;

            std::vector<double> o(m * n);

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

        std::vector<double> ravelBbox(const py::array_t<double> &bb)
        {
            auto _bb = bb.unchecked<2>();
            std::vector<double> result;

            for (pybind11::ssize_t i = 0; i < _bb.shape(0); i++)
            {
                result.push_back(_bb(i, 0));
                result.push_back(_bb(i, 1));
                result.push_back(_bb(i, 2));
                result.push_back(_bb(i, 3));
            }
            return result;
        }

        std::vector<double> rleIou(const std::vector<RLE> &dt, const std::vector<RLE> &gt, const uint64_t &m, const uint64_t &n, const std::vector<int> &iscrowd)
        {
            uint64_t g, d;
            std::vector<double> db, gb;
            int crowd;

            db = ravelBbox(rleToBbox(dt, m));
            gb = ravelBbox(rleToBbox(gt, n));

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
                            u = rleArea({dt[d]})[0];
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
                    return frUncompressedRLE(pyobj.cast<std::vector<py::dict>>(), h, w);
                }
                else if (sub_type == "<class 'list'>" or sub_type == "<class 'numpy.ndarray'>")
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
                else if (sub_type == "<class 'float'>" or sub_type == "<class 'int'>")
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
                return frUncompressedRLE({pyobj}, h, w)[0]; // need return first
            }
            else
            {
                throw py::type_error("input type is not supported.");
            }

            return _toString(rles);
        }

        std::variant<pybind11::dict, py::object> segmToRle(const py::object &pyobj, const uint64_t &w, const uint64_t &h)
        {
            std::string type = py::str(py::type::of(pyobj));
            if (type == "<class 'list'>")
            {
                std::vector<std::vector<double>> poly = pyobj.cast<std::vector<std::vector<double>>>();
                std::vector<RLE> rles;
                for (size_t i = 0; i < poly.size(); i++)
                {
                    rles.push_back(rleFrPoly(poly[i], poly[i].size() / 2, h, w));
                }
                return _toString({rleMerge(rles, 0)})[0];
            }
            else if (type == "<class 'dict'>")
            {
                return frUncompressedRLE({pyobj}, h, w)[0];
            }
            else
            {
                return pyobj;
            }
        }
    } // namespace Mask

}