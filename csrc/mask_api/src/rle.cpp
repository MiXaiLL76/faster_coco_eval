// Copyright (c) MiXaiLL76
#include "mask.h"
#include <time.h>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <iostream>
#include <execution>

using namespace pybind11::literals;

namespace mask_api
{
    namespace Mask
    {
        std::string RLE::toString() const
        {
            int64_t x;
            int64_t c;
            bool more;
            std::string result;

            for (uint64_t i = 0; i < this->m; i++)
            {
                x = (int64_t)this->cnts[i];
                if (i > 2)
                {
                    x -= (int64_t)this->cnts[i - 2];
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
            return result;
        }

        RLE RLE::frString(const std::string &s, const uint64_t &h, const uint64_t &w)
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

        std::vector<uint> RLE::toBbox() const
        {
            if (this->m == 0)
            {
                return {0, 0, 0, 0};
            }

            uint xs, ys, xe, ye, cc;

            size_t m = this->m & 1 ? this->m - 1 : this->m;
            uint h = (uint)this->h, w = (uint)this->w;

            xs = w;
            ys = h;
            xe = ye = 0;
            cc = 0;

            for (size_t j = 0; j < m; j++)
            {
                uint start = cc;     // start of current segment
                cc += this->cnts[j]; // start of next segment
                if (j % 2 == 0)
                    continue; // skip background segment
                if (this->cnts[j] == 0)
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
            return {xs, ys, (xe - xs + 1), (ye - ys + 1)};
        }
        RLE RLE::frPoly(const std::vector<double> &xy, const uint64_t &h, const uint64_t &w)
        {
            /* upsample and get discrete points densely along entire boundary */
            uint64_t j = 0;
            size_t k = xy.size() / 2;
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

        RLE RLE::frBbox(const std::vector<double> &bb, const uint64_t &h, const uint64_t &w)
        {
            double xs = bb[0], xe = xs + bb[2];
            double ys = bb[1], ye = ys + bb[3];
            return RLE::frPoly({xs, ys, xs, ye, xe, ye, xe, ys}, h, w);
        }

        RLE RLE::erode_3x3(int dilation) const
        {
            bool v = false;
            int64_t max_len = this->w * this->h;
            std::vector<bool> _counts(max_len, false);
            std::vector<bool>::iterator ptr = _counts.begin();
            std::for_each(this->cnts.begin(), this->cnts.end(), [&v, &ptr](uint count)
                          {
                
                if(v){
                    std::fill_n(ptr, count, v);
                }
                
                v = !v;
                ptr += count; });

            std::vector<int> ofsvec;
            std::vector<int> ofsvec_bottom;

            for (int i = dilation; i >= 0; i--)
            {
                for (int j = dilation; j >= -dilation; j--)
                {
                    if (i == 0 && j <= 0)
                    {
                        continue;
                    }
                    if (i > 0)
                    {
                        ofsvec.push_back(i * this->h + j);
                    }
                    else
                    {
                        ofsvec.push_back(j);
                    }
                }
            }

            for (int i = dilation; i >= -dilation; i--)
            {
                ofsvec_bottom.push_back(i * this->h + dilation);
            }

            int64_t c = 0;
            int64_t ic = 0;
            std::vector<uint> cnts;
            bool _min = false, _prev_min = false;
            uint64_t rle_h = this->h;

            v = true;
            for (uint j : this->cnts)
            {
                cnts.emplace_back(0);

                v = !v;
                if (v)
                {
                    _prev_min = false;

                    for (uint k = 0; k < j; k++)
                    {
                        if (c % rle_h != 0)
                        {
                            if (_prev_min)
                            {
                                _min = std::all_of(ofsvec_bottom.begin(), ofsvec_bottom.end(), [c, max_len, rle_h, &_counts](int o)
                                                   {
                                    int64_t test_ptr = c + o;
                                    return (
                                        (test_ptr > 0) && 
                                        (test_ptr < max_len) && _counts[test_ptr] && (test_ptr % rle_h != 0)
                                    ); });
                            }
                            else
                            {
                                _min = std::all_of(ofsvec.begin(), ofsvec.end(), [c, max_len, rle_h, &_counts](int o)
                                                   {
                                    int64_t test_ptr = c + o;
                                    int64_t _test_ptr = c - o;
                                    return (
                                        (_test_ptr > 0) && _counts[_test_ptr] && 
                                        (test_ptr < max_len) && _counts[test_ptr] && ((test_ptr + 1) % rle_h != 0)
                                    ); });
                            }
                        }
                        else
                        {
                            _min = false;
                        }

                        if (_min)
                        {
                            cnts[ic] += 1;
                        }
                        else
                        {
                            if (_prev_min)
                            {
                                cnts.insert(cnts.end(), {1, 0});
                                ic += 2;
                            }
                            else
                            {
                                cnts[ic - 1] += 1;
                            }
                        }
                        _prev_min = _min;

                        c++;
                    }
                }
                else
                {
                    cnts[ic] += j;
                    c += j;
                }

                ic++;
            }

            ic = 0;
            std::vector<uint> clean_cnts;
            bool last_zero = false;
            for (size_t i = 0; i < cnts.size(); i++)
            {
                if (i > 0)
                {
                    if (cnts[i] == 0 || last_zero)
                    {
                        clean_cnts[ic - 1] += cnts[i];
                    }
                    else
                    {
                        clean_cnts.emplace_back(cnts[i]);
                        ic++;
                    }
                }
                else
                {
                    clean_cnts.emplace_back(cnts[i]);
                    ic++;
                }
                last_zero = cnts[i] == 0;
            }

            return RLE(this->h, this->w, clean_cnts.size(), clean_cnts);
        }

        RLE RLE::merge(const std::vector<RLE> &R, const int &intersect)
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

                RLE B(0, 0, {});
                RLE A(h, w, m, {});

                for (i = 1; i < n; i++)
                {
                    B = R[i];
                    if (B.h != h || B.w != w)
                    {
                        h = w = m = 0;
                        break;
                    }
                    A = RLE(h, w, m, cnts);
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

        RLE RLE::toBoundary(double dilation_ratio) const
        {
            int dilation = (int)std::round((dilation_ratio * std::sqrt(this->h * this->h + this->w * this->w)) - 1e-10);
            if (dilation < 1)
            {
                dilation = 1;
            }
            return RLE::merge({*this, this->erode_3x3(dilation)}, -1);
        }

        uint RLE::area() const
        {
            uint result = 0;
            for (size_t j = 1; j < this->m; j += 2)
                result += this->cnts[j];

            return result;
        }

        py::dict RLE::toDict() const
        {
            return py::dict(
                "size"_a = std::vector<uint64_t>{this->h, this->w},
                "counts"_a = py::bytes(this->toString()));
        }

        std::tuple<uint64_t, uint64_t, std::string> RLE::toTuple() const
        {
            return std::tuple<uint64_t, uint64_t, std::string>{this->h, this->w, this->toString()};
        }

        RLE RLE::frTuple(const std::tuple<uint64_t, uint64_t, std::string> &w_h_rlestring)
        {
            return RLE::frString(std::get<2>(w_h_rlestring), std::get<0>(w_h_rlestring), std::get<1>(w_h_rlestring));
        }

        RLE RLE::frUncompressedRLE(const py::dict &ucRle)
        {
            std::pair<uint64_t, uint64_t> size = ucRle["size"].cast<std::pair<uint64_t, uint64_t>>();
            std::vector<uint> counts = ucRle["counts"].cast<std::vector<uint>>();
            return RLE(size.first, size.second, counts.size(), counts);
        }

        RLE RLE::frSegm(const py::object &pyobj, const uint64_t &w, const uint64_t &h)
        {
            std::string type = py::str(py::type::of(pyobj));
            if (type == "<class 'list'>")
            {
                std::vector<std::vector<double>> poly = pyobj.cast<std::vector<std::vector<double>>>();
                std::vector<RLE> rles;
                for (size_t i = 0; i < poly.size(); i++)
                {
                    rles.push_back(RLE::frPoly(poly[i], h, w));
                }
                return RLE::merge(rles, 0);
            }
            else if (type == "<class 'dict'>")
            {
                std::string sub_type = py::str(py::type::of(pyobj["counts"]));
                if (sub_type == "<class 'list'>")
                {
                    return RLE::frUncompressedRLE(pyobj);
                }
                else if (sub_type == "<class 'bytes'>" || sub_type == "<class 'str'>")
                {
                    std::pair<uint64_t, uint64_t> size = pyobj["size"].cast<std::pair<uint64_t, uint64_t>>();
                    std::string counts = pyobj["counts"].cast<std::string>();
                    return RLE::frString(counts, size.first, size.second);
                }
                else
                {
                    throw py::type_error("counts type mast be list for UncompressedRLE");
                }
            }
            else
            {
                throw py::type_error("type mast be list or dict for Segm");
            }
        }
    } // namespace Mask

}