// Copyright (c) MiXaiLL76
#include <time.h>

#include <algorithm>
#include <cstdint>
#include <execution>
#include <iostream>
#include <numeric>

#include "mask.h"

using namespace pybind11::literals;

template <typename T>
static bool AreEqual(T f1, T f2) {
        return (std::fabs(f1 - f2) <=
                std::numeric_limits<T>::epsilon() *
                    std::fmax(std::fabs(f1), std::fabs(f2)));
}

template <typename T>
void prinf_vector(const std::vector<T> vec, const std::string s) {
        std::cout << "name: " << s << std::endl;
        std::cout << "size: " << vec.size() << std::endl;

        for (const auto &v : vec) std::cout << "\t" << v << std::endl;

        std::cout << std::endl;
}

namespace mask_api {
namespace Mask {
// Converts the RLE object into a compact string representation.
// Each count is delta-encoded and variable-length encoded as a string.
std::string RLE::toString() const {
        std::string result;

        for (std::size_t i = 0; i < m; ++i) {
                int64_t x = static_cast<int64_t>(cnts[i]);
                // Apply delta encoding for all counts after the second entry
                if (i > 2) {
                        x -= static_cast<int64_t>(cnts[i - 2]);
                }

                bool more = true;
                // Variable-length encode the value
                while (more) {
                        int64_t c = x & 0x1f;  // Take 5 bits
                        x >>= 5;
                        // If the sign bit (0x10) is set, continue if x != -1;
                        // otherwise, continue if x != 0
                        more = (c & 0x10) ? x != -1 : x != 0;
                        if (more) c |= 0x20;  // Set continuation bit
                        c += 48;              // Shift to ASCII
                        result += static_cast<char>(c);
                }
        }
        return result;
}

// Converts an RLE-encoded string to an RLE object.
// Parameters:
//   - s: input string with RLE encoding
//   - h: mask height
//   - w: mask width
// Returns:
//   - RLE object representing the mask
RLE RLE::frString(const std::string &s, uint64_t h, uint64_t w) {
        std::vector<uint64_t> cnts;
        const std::size_t m = s.size();
        std::size_t i = 0;

        // Decode each run from the string
        while (i < m) {
                int64_t x = 0;
                int k = 0;
                bool more;

                do {
                        if (i >= m)
                                throw std::runtime_error(
                                    "RLE string is malformed: early end.");

                        // Subtract '0' to get value in range [0..63]
                        const int64_t c = static_cast<int64_t>(s[i]) -
                                          static_cast<int64_t>('0');
                        x |= (c & 0x1f) << (5 * k);
                        more = c & 0x20;
                        ++i;
                        ++k;

                        // If highest bit of this chunk is set and this is the
                        // last chunk, extend sign
                        if (!more && (c & 0x10)) {
                                x |= (~0LL) << (5 * k);
                        }
                } while (more);

                // Cumulative sum for elements after the second
                if (cnts.size() > 2) {
                        x += cnts[cnts.size() - 2];
                }

                cnts.emplace_back(static_cast<uint64_t>(x));
        }

        return RLE(h, w, cnts.size(), cnts);
}

// Returns the bounding box [x, y, w, h] for the RLE mask as a vector of
// doubles. The bounding box tightly encloses all foreground pixels in the mask.
std::vector<double> RLE::toBbox() const {
        // Handle empty mask
        if (this->m == 0) {
                return {0.0, 0.0, 0.0, 0.0};
        }

        uint64_t xs, ys, xe, ye, cc;

        // Only process even number of segments (if odd, ignore last)
        size_t m = (this->m & 1) ? this->m - 1 : this->m;
        uint64_t h = static_cast<uint64_t>(this->h),
                 w = static_cast<uint64_t>(this->w);

        xs = w;
        ys = h;
        xe = ye = 0;
        cc = 0;

        for (size_t j = 0; j < m; ++j) {
                uint64_t start = cc;  // Start index of current segment
                cc +=
                    this->cnts[j];  // End index (exclusive) of current segment

                if (j % 2 == 0) continue;  // Skip background segments

                if (this->cnts[j] == 0)
                        continue;  // Skip zero-length foreground segments

                uint64_t y_start = start % h, x_start = (start - y_start) / h;
                uint64_t y_end = (cc - 1) % h, x_end = (cc - 1 - y_end) / h;

                xs = std::min(xs, x_start);
                xe = std::max(xe, x_end);

                if (x_start < x_end) {
                        ys = 0;
                        ye = h - 1;  // Foreground segment goes across columns
                } else {
                        ys = std::min(ys, y_start);
                        ye = std::max(ye, y_end);
                }
        }
        // Return bounding box: [x, y, width, height]
        return {static_cast<double>(xs), static_cast<double>(ys),
                static_cast<double>(xe - xs + 1),
                static_cast<double>(ye - ys + 1)};
}

// Converts a polygon (represented by a list of (x, y) pairs) into an RLE mask
// representation. Parameters:
//   - xy: vector of double, length 2*k, representing (x, y) coordinates of
//   polygon vertices
//   - h: mask height
//   - w: mask width
// Returns:
//   - RLE object representing the binary mask of the polygon
RLE RLE::frPoly(const std::vector<double> &xy, uint64_t h, uint64_t w) {
        uint64_t j = 0;
        std::size_t k = xy.size() / 2;
        double scale = 5.0;

        std::vector<int> x(k + 1);
        std::vector<int> y(k + 1);

        // Upsample and get discrete points densely along the entire boundary
        for (j = 0; j < k; ++j) {
                x[j] = static_cast<int>(scale * xy[j * 2 + 0] + 0.5);
                y[j] = static_cast<int>(scale * xy[j * 2 + 1] + 0.5);
        }
        x[k] = x[0];
        y[k] = y[0];

        std::vector<int> u;
        std::vector<int> v;

        // Draw lines between consecutive points, using Bresenham-like approach
        for (j = 0; j < k; ++j) {
                int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1];
                int dx = std::abs(xe - xs);
                int dy = std::abs(ys - ye);
                int flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
                int t;
                double s;
                if (flip) {
                        std::swap(xs, xe);
                        std::swap(ys, ye);
                }
                s = dx >= dy ? static_cast<double>(ye - ys) / dx
                             : static_cast<double>(xe - xs) / dy;

                if (dx >= dy) {
                        for (int d = 0; d <= dx; ++d) {
                                t = flip ? dx - d : d;
                                u.emplace_back(t + xs);
                                v.emplace_back(
                                    static_cast<int>(ys + s * t + 0.5));
                        }
                } else {
                        for (int d = 0; d <= dy; ++d) {
                                t = flip ? dy - d : d;
                                v.emplace_back(t + ys);
                                u.emplace_back(
                                    static_cast<int>(xs + s * t + 0.5));
                        }
                }
        }

        // Get points along y-boundary and downsample
        x.clear();
        y.clear();
        double xd, yd;
        for (std::size_t j = 1; j < u.size(); ++j) {
                if (u[j] != u[j - 1]) {
                        xd = static_cast<double>(u[j] < u[j - 1] ? u[j]
                                                                 : u[j] - 1);
                        xd = (xd + 0.5) / scale - 0.5;
                        if ((!AreEqual(std::floor(xd), xd)) || xd < 0 ||
                            xd > static_cast<double>(w - 1)) {
                                continue;
                        }
                        yd = static_cast<double>(v[j] < v[j - 1] ? v[j]
                                                                 : v[j - 1]);
                        yd = (yd + 0.5) / scale - 0.5;
                        if (yd < 0)
                                yd = 0;
                        else if (yd > static_cast<double>(h))
                                yd = static_cast<double>(h);

                        yd = std::ceil(yd);
                        x.emplace_back(static_cast<int>(xd));
                        y.emplace_back(static_cast<int>(yd));
                }
        }

        // Compute RLE encoding given y-boundary points
        std::vector<uint32_t> a;
        for (std::size_t j = 0; j < x.size(); ++j)
                a.emplace_back(
                    static_cast<uint32_t>(x[j] * static_cast<int>(h) + y[j]));
        a.emplace_back(static_cast<uint32_t>(h * w));

        std::stable_sort(a.begin(), a.end());

        uint32_t p = 0;
        for (std::size_t j = 0; j < a.size(); ++j) {
                uint32_t t = a[j];
                a[j] -= p;
                p = t;
        }

        std::vector<uint64_t> b;
        std::size_t j2 = 1;
        b.emplace_back(a[0]);
        while (j2 < a.size()) {
                if (a[j2] > 0)
                        b.emplace_back(a[j2++]);
                else {
                        ++j2;
                        if (j2 < a.size()) b.back() += a[j2++];
                }
        }

        return RLE(h, w, b.size(), b);
}

// Converts an axis-aligned bounding box to an RLE mask representation.
// Parameters:
//   - bb: vector of 4 doubles, [x, y, width, height], representing the bounding
//   box
//   - h: mask height
//   - w: mask width
// Returns:
//   - RLE object representing the binary mask of the bounding box
RLE RLE::frBbox(const std::vector<double> &bb, uint64_t h, uint64_t w) {
        // Calculate the four corners of the rectangle
        double xs = bb[0], xe = bb[0] + bb[2];
        double ys = bb[1], ye = bb[1] + bb[3];

        // Construct the polygon in clockwise order and call frPoly
        return RLE::frPoly({xs, ys, xs, ye, xe, ye, xe, ys}, h, w);
}

// Performs morphological erosion with a 3x3 (or arbitrary dilation) structuring
// element on the binary mask represented by RLE. Parameters:
//   - dilation: size of the erosion (1 for 3x3, 2 for 5x5, etc.)
// Returns:
//   - Eroded RLE mask
RLE RLE::erode_3x3(int dilation) const {
        // Flatten RLE into a dense boolean mask
        long max_len = static_cast<long>(this->w * this->h);
        std::vector<bool> mask(max_len, false);

        bool v = false;
        uint64_t idx = 0;
        for (uint64_t cnt : this->cnts) {
                if (v) {
                        std::fill_n(mask.begin() + idx, cnt, true);
                }
                idx += cnt;
                v = !v;
        }

        // Prepare offsets for the structuring element (above and on the current
        // pixel)
        std::vector<int> ofsvec;
        std::vector<int> ofsvec_bottom;

        for (int i = dilation; i >= 0; i--) {
                for (int j = dilation; j >= -dilation; j--) {
                        if (i == 0 && j <= 0) continue;
                        if (i > 0)
                                ofsvec.push_back(i * static_cast<int>(this->h) +
                                                 j);
                        else
                                ofsvec.push_back(j);
                }
        }
        for (int i = dilation; i >= -dilation; i--) {
                ofsvec_bottom.push_back(i * static_cast<int>(this->h) +
                                        dilation);
        }

        // Erosion logic
        std::vector<uint64_t> cnts;
        long c = 0;
        size_t ic = 0;
        long rle_h = static_cast<long>(this->h);
        v = true;  // background is always first in RLE
        bool _min = false, _prev_min = false;

        for (uint64_t j : this->cnts) {
                cnts.emplace_back(0);
                v = !v;
                if (v) {
                        _prev_min = false;
                        for (uint64_t k = 0; k < j; k++) {
                                long y = c % rle_h;
                                if (_prev_min) {
                                        _min = std::all_of(
                                            ofsvec_bottom.begin(),
                                            ofsvec_bottom.end(),
                                            [c, max_len, rle_h, &mask, y,
                                             dilation](int o) {
                                                    long test_ptr = c + o;
                                                    return (test_ptr >= 0) &&
                                                           (test_ptr <
                                                            max_len) &&
                                                           mask[test_ptr] &&
                                                           (std::abs((test_ptr %
                                                                      rle_h) -
                                                                     y) <=
                                                            dilation);
                                            });
                                } else {
                                        _min = std::all_of(
                                            ofsvec.begin(), ofsvec.end(),
                                            [c, max_len, rle_h, &mask, y,
                                             dilation](int o) {
                                                    long test_ptr = c + o;
                                                    long test_ptr_mirror =
                                                        c - o;
                                                    return (test_ptr_mirror >=
                                                            0) &&
                                                           mask[test_ptr_mirror] &&
                                                           (test_ptr <
                                                            max_len) &&
                                                           mask[test_ptr] &&
                                                           (std::abs((test_ptr %
                                                                      rle_h) -
                                                                     y) <=
                                                            dilation);
                                            });
                                }

                                if (_min) {
                                        cnts[ic] += 1;
                                } else {
                                        if (_prev_min) {
                                                cnts.insert(cnts.end(), {1, 0});
                                                ic += 2;
                                        } else {
                                                if (ic > 0)  // Avoid underflow
                                                        cnts[ic - 1] += 1;
                                        }
                                }
                                _prev_min = _min;
                                ++c;
                        }
                } else {
                        cnts[ic] += j;
                        c += static_cast<long>(j);
                }
                ++ic;
        }

        // Remove duplicate runs and return
        return RLE(this->h, this->w, cnts.size(), cnts).clear_duplicates();
}

// Removes zero runs and merges consecutive duplicate runs in the RLE counts
// vector.
RLE RLE::clear_duplicates() const {
        std::vector<uint64_t> clean_cnts;
        bool last_zero = false;

        for (std::size_t i = 0; i < this->cnts.size(); ++i) {
                if (i > 0) {
                        if (this->cnts[i] == 0 || last_zero) {
                                // Merge zero and consecutive runs into the
                                // previous run
                                clean_cnts.back() += this->cnts[i];
                        } else {
                                clean_cnts.emplace_back(this->cnts[i]);
                        }
                } else {
                        clean_cnts.emplace_back(this->cnts[i]);
                }
                last_zero = (this->cnts[i] == 0);
        }

        return RLE(this->h, this->w, clean_cnts.size(), clean_cnts);
}

// Merges a list of RLE masks by union, intersection, or xor.
// Parameters:
//   - R: vector of RLE masks to merge (all must have same shape)
//   - intersect: 0 (union), 1 (intersection), 2 (xor)
// Returns:
//   - merged RLE mask, or empty RLE if incompatible input
RLE RLE::merge(const std::vector<RLE> &R, int intersect) {
        size_t n = R.size();

        if (n == 0) {
                return RLE(0, 0, 0, {});
        }
        if (n == 1) {
                return R[0];
        }

        // All masks must have the same shape
        uint64_t h = R[0].h, w = R[0].w;
        size_t max_len = w * h;
        for (size_t i = 1; i < n; ++i) {
                if (R[i].h != h || R[i].w != w) {
                        return RLE(0, 0, 0, {});
                }
        }

        // Decode the first mask into a dense boolean vector
        std::vector<bool> mask(max_len, false);
        bool v = false;
        size_t idx = 0;
        for (uint64_t cnt : R[0].cnts) {
                if (v) {
                        std::fill_n(mask.begin() + idx, cnt, true);
                }
                idx += cnt;
                v = !v;
        }

        // Merge all other masks
        for (size_t i = 1; i < n; ++i) {
                v = false;
                size_t cc = 0;
                for (uint64_t cnt : R[i].cnts) {
                        for (size_t j = cc; j < cc + cnt; ++j) {
                                if (intersect == 0) {
                                        mask[j] = mask[j] | v;  // union
                                } else if (intersect == 1) {
                                        mask[j] = mask[j] & v;  // intersection
                                } else {
                                        mask[j] = mask[j] ^ v;  // xor
                                }
                        }
                        v = !v;
                        cc += cnt;
                }
        }

        // Re-encode to RLE
        std::vector<uint64_t> out_cnts;
        v = false;
        uint64_t run_len = 0;
        for (size_t i = 0; i < max_len; ++i) {
                if (mask[i] != v) {
                        out_cnts.push_back(run_len);
                        run_len = 1;
                        v = !v;
                } else {
                        ++run_len;
                }
        }
        out_cnts.push_back(run_len);

        return RLE(h, w, out_cnts.size(), out_cnts);
}

// Returns the boundary of the mask as a new RLE, using morphological erosion
// and XOR. Parameters:
//   - dilation_ratio: ratio of mask diagonal for erosion radius (typical:
//   0.008)
// Returns:
//   - RLE mask representing the boundary pixels
RLE RLE::toBoundary(double dilation_ratio) const {
        // Compute dilation size as a function of mask diagonal and requested
        // ratio
        int dilation = static_cast<int>(std::round(
            dilation_ratio * std::sqrt(this->h * this->h + this->w * this->w) -
            1e-10));
        if (dilation < 1) {
                dilation = 1;
        }
        // Merge original and eroded mask using XOR (intersect = -1)
        return RLE::merge({*this, this->erode_3x3(dilation)}, -1);
}

// Returns the total area (number of foreground pixels) represented by the RLE
// mask. Only the odd-indexed runs (foreground) are summed.
uint64_t RLE::area() const {
        uint64_t result = 0;
        for (std::size_t j = 1; j < this->m; j += 2) result += this->cnts[j];
        return result;
}

// Assuming RLE has .h, .w, and .toString() methods.
pybind11::dict RLE::toDict() const {
        namespace py = pybind11;
        return py::dict("size"_a = std::vector<uint64_t>{this->h, this->w},
                        "counts"_a = py::bytes(this->toString()));
}

// Returns a tuple containing (height, width, RLE string)
std::tuple<uint64_t, uint64_t, std::string> RLE::toTuple() const {
        return std::tuple<uint64_t, uint64_t, std::string>{this->h, this->w,
                                                           this->toString()};
}

// Constructs an RLE object from a tuple (height, width, rle_string)
RLE RLE::frTuple(
    const std::tuple<uint64_t, uint64_t, std::string> &w_h_rlestring) {
        return RLE::frString(std::get<2>(w_h_rlestring),  // rle_string
                             std::get<0>(w_h_rlestring),  // height
                             std::get<1>(w_h_rlestring)   // width
        );
}

// Constructs an RLE object from an uncompressed RLE Python dictionary
RLE RLE::frUncompressedRLE(const pybind11::dict &ucRle) {
        // Extract size as a pair (height, width)
        std::pair<uint64_t, uint64_t> size =
            ucRle["size"].cast<std::pair<uint64_t, uint64_t>>();
        // Extract counts as a vector of uints
        std::vector<uint64_t> counts =
            ucRle["counts"].cast<std::vector<uint64_t>>();
        // Construct and return the RLE
        return RLE(size.first, size.second, counts.size(), counts);
}

// Constructs an RLE object from a segmentation input (polygon or RLE dict).
RLE RLE::frSegm(const pybind11::object &pyobj, uint64_t w, uint64_t h) {
        namespace py = pybind11;
        std::string type = py::str(py::type::of(pyobj));

        if (type == "<class 'list'>") {
                // Polygon format: list of polygons
                std::vector<std::vector<double>> poly =
                    pyobj.cast<std::vector<std::vector<double>>>();
                std::vector<RLE> rles;
                for (const auto &p : poly) {
                        rles.push_back(RLE::frPoly(p, h, w));
                }
                return RLE::merge(rles, 0);  // union of polygons
        } else if (type == "<class 'dict'>") {
                std::string sub_type =
                    py::str(py::type::of(pyobj.attr("get")("counts")));
                if (sub_type == "<class 'list'>") {
                        // Uncompressed RLE
                        return RLE::frUncompressedRLE(pyobj);
                } else if (sub_type == "<class 'bytes'>" ||
                           sub_type == "<class 'str'>") {
                        // Compressed RLE
                        std::pair<uint64_t, uint64_t> size =
                            pyobj["size"].cast<std::pair<uint64_t, uint64_t>>();
                        std::string counts =
                            pyobj["counts"].cast<std::string>();
                        return RLE::frString(counts, size.first, size.second);
                } else {
                        throw py::type_error(
                            "counts type must be list, bytes, or str for RLE");
                }
        } else {
                throw py::type_error(
                    "Segm type must be list (polygon) or dict (RLE)");
        }
}
}  // namespace Mask

}  // namespace mask_api
