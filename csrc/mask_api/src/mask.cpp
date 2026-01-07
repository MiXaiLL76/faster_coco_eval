// Copyright (c) MiXaiLL76
#if defined(_MSC_VER)
#include <cstddef>
typedef std::ptrdiff_t ssize_t;
#else
#include <sys/types.h>
#endif

#include <time.h>

#include <algorithm>
#include <cstdint>
#include <execution>
#include <future>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#include "mask.h"

using namespace pybind11::literals;

namespace mask_api {

namespace Mask {

// Converts an RLE object to a Python bytes object using its toString() method.
py::bytes rleToString(const RLE& R) { return py::bytes(R.toString()); }

// Simple wrapper for RLE::frString
RLE rleFrString(const std::string& s, const uint64_t& h, const uint64_t& w) {
        return RLE::frString(s, h, w);
}

// Encodes a 3D numpy array (py::array_t<uint8_t, py::array::f_style>) into a
// vector of RLE (Run-Length Encoding) objects. Parameters:
//   M - A 3D numpy array with dimensions [height, width, n], containing uint8_t
//   mask data. h - The height of the mask (number of rows). w - The width of
//   the mask (number of columns). n - The number of masks (third dimension of
//   M).
//
// Returns:
//   A std::vector of RLE objects, each representing the run-length encoding of
//   a single mask.
//
// The function iterates through each mask (along the third dimension), and for
// each mask, it traverses all elements in column-major order. It counts
// consecutive runs of identical values and stores the counts in a vector, which
// is then used to construct an RLE object.
std::vector<RLE> rleEncode(const py::array_t<uint8_t, py::array::f_style>& M,
                           uint64_t h, uint64_t w, uint64_t n) {
        auto mask = M.unchecked<3>();

        std::vector<RLE> rles;
        rles.reserve(n);

        for (uint64_t i = 0; i < n; ++i) {
                std::vector<uint64_t> cnts;
                // Improved allocation strategy: adaptive sizing based on mask
                // characteristics
                size_t min_reserve = 16;  // Minimum reasonable size
                size_t max_reserve =
                    h * w / 4;  // Maximum for very complex masks
                size_t perimeter_estimate =
                    2 * (h + w);  // Typical perimeter-based estimate
                size_t estimated_size = std::max(
                    min_reserve, std::min(max_reserve, perimeter_estimate));
                cnts.reserve(estimated_size);

                uint8_t prev = 0;
                uint64_t count = 0;

                // Traverse the mask in column-major order
                for (uint64_t row = 0; row < w; ++row) {
                        for (uint64_t col = 0; col < h; ++col) {
                                uint8_t value = mask(col, row, i);
                                if (value != prev) {
                                        cnts.emplace_back(count);
                                        count = 0;
                                        prev = value;
                                }
                                ++count;
                        }
                }
                cnts.emplace_back(count);

                rles.emplace_back(h, w, cnts.size(), std::move(cnts));
        }
        return rles;
}

// Converts a vector of RLE (Run-Length Encoding) objects to a 2D NumPy array of
// bounding boxes. Parameters:
//   R - A std::vector of RLE objects. Each RLE object should provide a toBbox()
//   method that returns a vector of 4 unsigned integers
//       representing the bounding box in the format [x_min, y_min, width,
//       height].
//   n - (optional) The number of RLE elements to process. If not provided, all
//   elements in R are processed.
//
// Returns:
//   A py::array NumPy array of shape [n, 4], where each row corresponds to the
//   bounding box of a mask.
//
// The function processes up to n RLE objects (or all if n is not specified).
// For each RLE object, it extracts the bounding box using the toBbox() method
// and appends the results to a flat std::vector<double>. Finally, the function
// returns a NumPy array of shape [n, 4] containing all bounding boxes.
py::array rleToBbox(const std::vector<RLE>& R, std::optional<uint64_t> n) {
        size_t count = n.value_or(R.size());

        // Create py::array_t with proper memory management using shape vector
        std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(count), 4};
        py::array_t<double> result(shape);

        auto buf = result.mutable_unchecked<2>();

        for (size_t i = 0; i < count && i < R.size(); ++i) {
                auto bbox = R[i].toBbox();
                for (size_t j = 0; j < 4 && j < bbox.size(); ++j) {
                        buf(i, j) = bbox[j];
                }
        }

        return result;
}

// Assumes _frString and rleToBbox are defined elsewhere and compatible.
py::array_t<double> toBbox(const std::vector<py::dict>& R) {
        std::vector<RLE> rles = _frString(R);
        return rleToBbox(rles, rles.size());
}

// Converts a flat vector of bounding box coordinates to a vector of RLE
// (Run-Length Encoding) objects representing binary masks. Parameters:
//   bb - A std::vector<double> containing bounding box coordinates. Each
//   bounding box is represented by 4 consecutive values:
//        [x_min, y_min, width, height].
//   h  - The height of the image as an unsigned 64-bit integer.
//   w  - The width of the image as an unsigned 64-bit integer.
//   n  - The number of bounding boxes to process.
//
// Returns:
//   A std::vector<RLE> where each element is an RLE object corresponding to one
//   bounding box, encoded as a binary mask using the given height and width.
//
// The function checks if the input vector bb has at least n*4 elements. For
// each bounding box, it extracts its four coordinates, constructs a bounding
// box array, and passes it to RLE::frBbox to create the RLE mask. All resulting
// RLE objects are collected in a std::vector<RLE> which is returned to the
// caller. Throws std::invalid_argument if the input vector is too small.
std::vector<RLE> rleFrBbox(const std::vector<double>& bb, uint64_t h,
                           uint64_t w, uint64_t n) {
        std::vector<RLE> result;
        result.reserve(n);

        // Check if the input vector has enough elements for n bounding boxes
        if (bb.size() < n * 4) {
                throw std::invalid_argument(
                    "Input vector bb has insufficient size");
        }

        // Reuse single bbox vector to minimize allocations (4 doubles = 32
        // bytes)
        std::vector<double> bbox(4);

        for (uint64_t i = 0; i < n; ++i) {
                // Efficiently copy 4 consecutive values without temporary
                // object creation
                size_t base_idx = i * 4;
                bbox[0] = bb[base_idx + 0];
                bbox[1] = bb[base_idx + 1];
                bbox[2] = bb[base_idx + 2];
                bbox[3] = bb[base_idx + 3];

                // Convert the bounding box to RLE and add to the result
                result.emplace_back(RLE::frBbox(bbox, h, w));
        }
        return result;
}

// Simple wrapper for RLE::frPoly, ignoring parameter k and forwarding xy, h, w
RLE rleFrPoly(const std::vector<double>& xy, const uint64_t& k,
              const uint64_t& h, const uint64_t& w) {
        // 'k' is unused, kept for compatibility with calling convention
        return RLE::frPoly(xy, h, w);
}

// Converts a vector of RLE objects to a vector of Python dicts using
// RLE::toDict().
std::vector<py::dict> _toString(const std::vector<RLE>& rles) {
        std::vector<py::dict> result;
        result.reserve(rles.size());
        for (const auto& rle : rles) {
                result.push_back(rle.toDict());
        }
        return result;
}

// internal conversion from compressed RLE format to Python RLEs object
std::vector<RLE> _frString(const std::vector<py::dict>& R) {
        std::vector<RLE> result;
        for (uint64_t i = 0; i < R.size(); i++) {
                std::pair<uint64_t, uint64_t> size =
                    R[i]["size"].cast<std::pair<uint64_t, uint64_t>>();
                std::string counts = R[i]["counts"].cast<std::string>();
                result.emplace_back(
                    RLE::frString(counts, size.first, size.second));
        }
        return result;
}

std::vector<py::dict> encode(
    const py::array_t<uint8_t, py::array::f_style>& M) {
        return _toString(rleEncode(M, M.shape(0), M.shape(1), M.shape(2)));
}

// Converts a set of RLE-encoded binary masks into a single contiguous 3D array,
// where each mask occupies a separate slice along the third axis. Throws an
// exception if the encoding is invalid or exceeds the expected output
// dimensions.
//
// Parameters:
//  - const std::vector<RLE>& R: A vector of RLE structures, each representing a
//  binary mask with specific height (h),
//    width (w), a count of RLE pairs (m), and a vector of run counts (cnts).
//
// Returns:
//  - py::array_t<uint8_t, py::array::f_style>: A 3D numpy array with shape [h,
//  w, n], where n is the number of masks.
//    Each mask is stored as a 2D binary array in the output along the third
//    dimension.
//
// Usage:
//  std::vector<RLE> rle_masks = ...;
//  py::array_t<uint8_t, py::array::f_style> masks = rleDecode(rle_masks);
py::array_t<uint8_t, py::array::f_style> rleDecode(const std::vector<RLE>& R) {
        if (R.empty()) return {};

        uint64_t h = R[0].h;
        uint64_t w = R[0].w;
        size_t n = R.size();
        uint64_t s = h * w * n;

        py::array_t<uint8_t, py::array::f_style> M(
            {static_cast<size_t>(h), static_cast<size_t>(w), n});
        auto mask = M.mutable_unchecked<3>();

        for (size_t i = 0; i < n; ++i) {
                uint8_t v = 0;
                uint64_t x = 0, y = 0, c = 0;

                for (uint64_t j = 0; j < R[i].m; ++j) {
                        for (uint64_t k = 0; k < R[i].cnts[j]; ++k) {
                                if (c >= s) {
                                        std::stringstream ss;
                                        ss << "Invalid RLE mask "
                                              "representation; out of range "
                                              "HxW=[0;0]->["
                                           << h - 1 << ";" << w - 1
                                           << "] x=" << x << "; y=" << y;
                                        throw std::range_error(ss.str());
                                }

                                mask(y, x, i) = v;
                                ++c;
                                ++y;
                                if (y >= h) {
                                        y = 0;
                                        ++x;
                                }
                        }
                        v = !v;
                }
        }
        return M;
}

// decode mask from compressed list of RLE string or RLEs object
py::array_t<uint8_t, py::array::f_style> decode(
    const std::vector<py::dict>& R) {
        return rleDecode(_frString(R));
}

std::vector<py::dict> erode_3x3(const std::vector<py::dict>& rleObjs,
                                const int& dilation) {
        std::vector<RLE> rles = _frString(rleObjs);
        std::transform(
            rles.begin(), rles.end(), rles.begin(),
            [dilation](const RLE& rle) { return rle.erode_3x3(dilation); });
        return _toString(rles);
}

std::vector<py::dict> toBoundary(const std::vector<py::dict>& rleObjs,
                                 const double& dilation_ratio = 0.02) {
        std::vector<RLE> rles = _frString(rleObjs);
        std::transform(rles.begin(), rles.end(), rles.begin(),
                       [&dilation_ratio](RLE const& rle) {
                               return rle.toBoundary(dilation_ratio);
                       });

        return _toString(rles);
}

py::dict merge(const std::vector<py::dict>& rleObjs, const int& intersect = 0) {
        return _toString({RLE::merge(_frString(rleObjs), intersect)})[0];
}
py::dict merge(const std::vector<py::dict>& rleObjs) {
        return merge(rleObjs, 0);
}

py::array_t<uint64_t> area(const std::vector<py::dict>& rleObjs) {
        std::vector<RLE> rles = _frString(rleObjs);
        std::vector<uint64_t> areas(rles.size());
        std::transform(rles.begin(), rles.end(), areas.begin(),
                       [](RLE const& rle) { return rle.area(); });
        return py::array(areas.size(), areas.data());
}

std::vector<py::dict> frPoly(const std::vector<std::vector<double>>& poly,
                             const uint64_t& h, const uint64_t& w) {
        std::vector<RLE> rles;
        for (uint64_t i = 0; i < poly.size(); i++) {
                rles.emplace_back(RLE::frPoly(poly[i], h, w));
        }
        return _toString(rles);
}

std::vector<py::dict> frBbox(const std::vector<std::vector<double>>& bb,
                             const uint64_t& h, const uint64_t& w) {
        std::vector<RLE> rles;
        rles.reserve(bb.size());  // Reserve memory for efficiency

        // Convert each bounding box to an RLE object
        for (const auto& box : bb) {
                rles.emplace_back(RLE::frBbox(box, h, w));
        }
        return _toString(rles);
}

std::vector<py::dict> rleToUncompressedRLE(const std::vector<RLE>& R) {
        std::vector<py::dict> result;
        for (uint64_t i = 0; i < R.size(); i++) {
                std::vector<uint64_t> size = {R[i].h, R[i].w};
                result.push_back(
                    py::dict("size"_a = size, "counts"_a = R[i].cnts));
        }
        return result;
}

std::vector<py::dict> toUncompressedRLE(const std::vector<py::dict>& Rles) {
        return rleToUncompressedRLE(_frString(Rles));
}

std::vector<py::dict> frUncompressedRLE(const std::vector<py::dict>& ucRles) {
        std::vector<RLE> rles;
        for (uint64_t i = 0; i < ucRles.size(); i++) {
                std::pair<uint64_t, uint64_t> size =
                    ucRles[i]["size"].cast<std::pair<uint64_t, uint64_t>>();
                std::vector<uint64_t> counts =
                    ucRles[i]["counts"].cast<std::vector<uint64_t>>();
                rles.emplace_back(size.first, size.second, counts.size(),
                                  counts);
        }
        return _toString(rles);
}

// Calculates the Intersection over Union (IoU) between two sets of bounding
// boxes.
//
// Parameters:
//   - dt: vector of doubles, concatenated predicted boxes, each as [x, y, w, h]
//   - gt: vector of doubles, concatenated ground-truth boxes, each as [x, y, w,
//   h]
//   - m: number of predicted (dt) boxes
//   - n: number of ground-truth (gt) boxes
//   - iscrowd: vector of int, with n entries; if iscrowd[g] is set, special
//   crowd IoU computation is used
//
// Returns:
//   - vector<double>: m*n IoU values between every dt and gt box (row-major:
//   o[d*n + g])
std::vector<double> bbIou(const std::vector<double>& dt,
                          const std::vector<double>& gt, std::size_t m,
                          std::size_t n, const std::vector<int>& iscrowd) {
        std::vector<double> o(m * n, 0.0);

        // Optional: check input sizes for early exit or error
        if (dt.size() != m * 4 || gt.size() != n * 4)
                throw std::invalid_argument(
                    "Input box vector size does not match m or n.");

        if (!iscrowd.empty() && iscrowd.size() != n)
                throw std::invalid_argument(
                    "iscrowd size must be 0 or equal to n.");

        const bool useCrowd = !iscrowd.empty();

        for (std::size_t g = 0; g < n; ++g) {
                const std::size_t offset_gt = g * 4;
                const double gx = gt[offset_gt + 0];
                const double gy = gt[offset_gt + 1];
                const double gw = gt[offset_gt + 2];
                const double gh = gt[offset_gt + 3];
                const double ga = gw * gh;
                const bool crowd = useCrowd && static_cast<bool>(iscrowd[g]);

                for (std::size_t d = 0; d < m; ++d) {
                        const std::size_t offset_dt = d * 4;
                        const double dx = dt[offset_dt + 0];
                        const double dy = dt[offset_dt + 1];
                        const double dw = dt[offset_dt + 2];
                        const double dh = dt[offset_dt + 3];
                        const double da = dw * dh;

                        // compute intersection
                        const double intersect_w =
                            std::min(dx + dw, gx + gw) - std::max(dx, gx);
                        if (intersect_w <= 0.0) continue;
                        const double intersect_h =
                            std::min(dy + dh, gy + gh) - std::max(dy, gy);
                        if (intersect_h <= 0.0) continue;
                        const double inter = intersect_w * intersect_h;
                        const double uni = crowd ? da : da + ga - inter;
                        o[d * n + g] = inter / uni;
                }
        }
        return o;
}

std::vector<double> rleIou(const std::vector<RLE>& dt,
                           const std::vector<RLE>& gt, const uint64_t& m,
                           const uint64_t& n, const std::vector<int>& iscrowd) {
        uint64_t g, d;
        std::vector<double> db, gb;
        int crowd;

        // Pre-allocate memory to avoid reallocations
        db.reserve(m * 4);
        gb.reserve(n * 4);

        // Avoid intermediate vector creation and double copying
        for (uint64_t i = 0; i < m; i++) {
                const auto bbox = dt[i].toBbox();
                db.insert(db.end(), bbox.begin(), bbox.end());
        }
        for (uint64_t i = 0; i < n; i++) {
                const auto bbox = gt[i].toBbox();
                gb.insert(gb.end(), bbox.begin(), bbox.end());
        }

        std::vector<double> o = bbIou(db, gb, m, n, iscrowd);
        bool _iscrowd = iscrowd.size() > 0;

        for (g = 0; g < n; g++) {
                for (d = 0; d < m; d++) {
                        if (o[d * n + g] > 0) {
                                crowd = _iscrowd && iscrowd[g];
                                if (dt[d].h != gt[g].h || dt[d].w != gt[g].w) {
                                        o[g * n + d] = -1;
                                        continue;
                                }
                                uint64_t ka, kb, a, b, c, ca, cb, ct, i, u;

                                int va, vb;
                                ca = dt[d].cnts[0];
                                ka = dt[d].m;
                                va = vb = 0;
                                cb = gt[g].cnts[0];
                                kb = gt[g].m;
                                a = b = 1;
                                i = u = 0;
                                ct = 1;
                                while (ct > 0) {
                                        c = std::min(ca, cb);
                                        if (va || vb) {
                                                u += c;
                                                if (va && vb) i += c;
                                        }
                                        ct = 0;
                                        ca -= c;
                                        if (!ca && a < ka) {
                                                ca = dt[d].cnts[a++];
                                                va = !va;
                                        }
                                        ct += ca;
                                        cb -= c;
                                        if (!cb && b < kb) {
                                                cb = gt[g].cnts[b++];
                                                vb = !vb;
                                        }
                                        ct += cb;
                                }
                                if (i == 0)
                                        u = 1;
                                else if (crowd) {
                                        u = dt[d].area();
                                }
                                o[d * n + g] = (double)i / (double)u;
                        }
                }
        }
        return o;
}

// Converts a Python object representing bounding boxes into a flat 1D vector of
// doubles. Parameters:
//   - pyobj: Python object (numpy.ndarray or list of lists) with shape [N][4].
// Returns:
//   - 1D vector of bounding box coordinates (size: N*4).
// Throws:
//   - std::out_of_range if input is not of shape Nx4.
std::vector<double> _preproc_bbox_array(const py::object& pyobj) {
        // Try to cast directly to numpy array for better performance
        if (py::isinstance<py::array_t<double>>(pyobj)) {
                auto arr = pyobj.cast<py::array_t<double>>();
                if (arr.ndim() == 2 && arr.shape(1) == 4) {
                        auto buf = arr.unchecked<2>();
                        std::vector<double> result;
                        result.reserve(arr.shape(0) * 4);

                        for (py::ssize_t i = 0; i < arr.shape(0); ++i) {
                                for (py::ssize_t j = 0; j < 4; ++j) {
                                        result.emplace_back(buf(i, j));
                                }
                        }
                        return result;
                }
                throw std::out_of_range(
                    "numpy ndarray input is only for *bounding boxes* and "
                    "should have "
                    "Nx4 dimension");
        }

        // Fallback to vector<vector<double>> for other types
        auto array = pyobj.cast<std::vector<std::vector<double>>>();
        if (array.empty()) {
                return std::vector<double>{};
        }

        if (array[0].size() != 4) {
                throw std::out_of_range(
                    "numpy ndarray input is only for *bounding boxes* and "
                    "should have "
                    "Nx4 dimension");
        }

        // Optimize: directly create flat vector
        std::vector<double> result;
        result.reserve(array.size() * 4);

        for (const auto& bbox : array) {
                if (bbox.size() != 4) {
                        throw std::out_of_range(
                            "numpy ndarray input is only for *bounding boxes* "
                            "and should have "
                            "Nx4 dimension");
                }
                result.insert(result.end(), bbox.begin(), bbox.end());
        }

        return result;
}

// Preprocesses a Python object into a flat vector of bounding boxes or a vector
// of RLEs. Parameters:
//   - pyobj: Python object, can be numpy.ndarray (boxes), list of lists
//   (boxes), or list of dicts (RLEs).
// Returns:
//   - Tuple containing:
//       - variant (vector<double> for boxes, vector<RLE> for RLEs)
//       - number of items (boxes or RLEs).
// Throws:
//   - std::out_of_range if the input type is unsupported or malformed.
std::tuple<std::variant<std::vector<RLE>, std::vector<double>>, size_t>
_preproc(const py::object& pyobj) {
        std::string type = py::str(py::type::of(pyobj));
        if (type == "<class 'numpy.ndarray'>") {
                auto result = _preproc_bbox_array(pyobj);
                return {result, result.size() / 4};
        } else if (type == "<class 'list'>") {
                auto pyobj_list = pyobj.cast<std::vector<py::object>>();
                if (pyobj_list.empty()) {
                        return {std::vector<double>{}, 0};
                }
                std::string sub_type = py::str(py::type::of(pyobj_list[0]));
                if (sub_type == "<class 'list'>" ||
                    sub_type == "<class 'numpy.ndarray'>") {
                        auto matrix =
                            pyobj.cast<std::vector<std::vector<double>>>();
                        for (const auto& item : matrix) {
                                if (item.size() != 4) {
                                        goto check_rle;
                                }
                        }
                        // Optimize: directly create flat vector without
                        // flatten() call
                        std::vector<double> result;
                        result.reserve(matrix.size() * 4);
                        for (const auto& bbox : matrix) {
                                result.insert(result.end(), bbox.begin(),
                                              bbox.end());
                        }
                        return {result, result.size() / 4};
                }
        check_rle:
                if (sub_type == "<class 'dict'>") {
                        auto result =
                            _frString(pyobj.cast<std::vector<py::dict>>());
                        return {result, result.size()};
                }
                throw std::out_of_range(
                    "list input can be bounding box (Nx4) or RLEs ([RLE])");
        }
        throw std::out_of_range(
            "unrecognized type. Supported types: RLEs (rle), np.ndarray (box), "
            "and list (box/RLE).");
}

// Computes the Intersection over Union (IoU) between detections and ground
// truths. Supports both bounding box and RLE inputs, and checks input
// compatibility. Parameters:
//   - dt: Python object representing detections (boxes or RLEs).
//   - gt: Python object representing ground truths (boxes or RLEs).
//   - iscrowd: Vector<int> indicating crowd regions in the ground truths.
// Returns:
//   - 2D numpy array (shape: [m, n]) of IoU values, where m=number of
//   detections and n=number of gt objects.
//   - If no matches, returns an empty vector<double>.
// Throws:
//   - std::out_of_range if types differ or iscrowd length mismatches gt count.
std::variant<py::array_t<double, py::array::f_style>, std::vector<double>> iou(
    const py::object& dt, const py::object& gt,
    const std::vector<int>& iscrowd) {
        auto [_dt, m] = _preproc(dt);
        auto [_gt, n] = _preproc(gt);

        if (m == 0 || n == 0) {
                return std::vector<double>{};
        }
        if (_dt.index() != _gt.index()) {
                throw std::out_of_range(
                    "The dt and gt should have the same data type, either "
                    "RLEs, list or np.ndarray");
        }
        if (!iscrowd.empty() && iscrowd.size() != n) {
                throw std::out_of_range(
                    "iscrowd must have the same length as gt");
        }

        std::vector<double> iou_result;
        if (std::holds_alternative<std::vector<double>>(_dt)) {
                const auto& _dt_box = std::get<std::vector<double>>(_dt);
                const auto& _gt_box = std::get<std::vector<double>>(_gt);
                iou_result = bbIou(_dt_box, _gt_box, m, n, iscrowd);
        } else {
                const auto& _dt_rle = std::get<std::vector<RLE>>(_dt);
                const auto& _gt_rle = std::get<std::vector<RLE>>(_gt);
                iou_result = rleIou(_dt_rle, _gt_rle, m, n, iscrowd);
        }
        return py::array(iou_result.size(), iou_result.data()).reshape({m, n});
}

// Converts a Python object representing segmentation data into COCO RLE or
// polygon/bbox encodings as pybind11 dicts. Handles lists, numpy arrays, and
// dicts, and dispatches to the appropriate encoding function based on the
// object type and shape. Parameters:
//   - pyobj: Python object representing the input segmentation (could be a
//   list, numpy.ndarray, or dict).
//   - h: Image height (uint64_t), used for encoding.
//   - w: Image width (uint64_t), used for encoding.
// Returns:
//   - If input is a list of objects, returns a vector of pybind11::dict
//   representing encoded masks or polygons.
//   - If input is a single object (dict or bbox/poly array), returns a single
//   pybind11::dict representing the encoded mask or polygon.
// Throws:
//   - std::out_of_range if the input list is empty or has invalid shape.
//   - py::type_error if the input type is not supported.
std::variant<pybind11::dict, std::vector<pybind11::dict>> frPyObjects(
    const py::object& pyobj, const uint64_t& h, const uint64_t& w) {
        std::vector<RLE> rles;
        std::string type = py::str(py::type::of(pyobj));

        // Handle Python list input
        if (type == "<class 'list'>") {
                std::vector<py::object> pyobj_list =
                    pyobj.cast<std::vector<py::object>>();
                if (pyobj_list.size() == 0) {
                        throw std::out_of_range("list index out of range");
                }

                std::string sub_type = py::str(py::type::of(pyobj_list[0]));

                // List of dicts: treat as uncompressed RLEs
                if (sub_type == "<class 'dict'>") {
                        return frUncompressedRLE(
                            pyobj.cast<std::vector<py::dict>>());
                }
                // List of lists or numpy arrays: treat as bbox or polygon
                // depending on shape
                else if ((sub_type == "<class 'list'>") ||
                         (sub_type == "<class 'numpy.ndarray'>")) {
                        std::vector<std::vector<double>> numpy_array =
                            pyobj.cast<std::vector<std::vector<double>>>();
                        if (numpy_array[0].size() == 4) {
                                return frBbox(numpy_array, h, w);
                        } else if (numpy_array[0].size() > 4) {
                                return frPoly(numpy_array, h, w);
                        }
                }
                // List of floats or ints: treat as a single bbox or polygon
                // depending on length
                else if ((sub_type == "<class 'float'>") ||
                         (sub_type == "<class 'int'>")) {
                        std::vector<double> array =
                            pyobj.cast<std::vector<double>>();
                        if (array.size() == 4) {
                                return frBbox(
                                    {array}, h,
                                    w)[0];  // Return the first (and only) dict
                        } else if (array.size() > 4) {
                                return frPoly(
                                    {array}, h,
                                    w)[0];  // Return the first (and only) dict
                        }
                }
        }
        // Handle numpy ndarray input as bounding boxes
        else if (type == "<class 'numpy.ndarray'>") {
                return frBbox(pyobj.cast<std::vector<std::vector<double>>>(), h,
                              w);
        }
        // Handle single dict input as uncompressed RLE
        else if (type == "<class 'dict'>") {
                return frUncompressedRLE(
                    {pyobj})[0];  // Return the first (and only) dict
        } else {
                throw py::type_error("input type is not supported.");
        }

        return _toString(rles);
}

// Converts a segmentation Python object to a COCO RLE (Run-Length Encoding)
// dictionary. If conversion is not possible due to a type error, returns the
// original Python object. Parameters:
//   - pyobj: Python object representing the segmentation mask (polygon,
//   uncompressed RLE, or other compatible format).
//   - w: Image width (uint64_t), used for encoding.
//   - h: Image height (uint64_t), used for encoding.
// Returns:
//   - pybind11::dict containing the RLE-encoded mask if conversion succeeds.
//   - Otherwise, returns the original py::object if conversion fails with
//   py::type_error.
std::variant<pybind11::dict, py::object> segmToRle(const py::object& pyobj,
                                                   const uint64_t& w,
                                                   const uint64_t& h) {
        try {
                RLE rle = RLE::frSegm(pyobj, w, h);
                return rle.toDict();
        } catch (py::type_error const&) {
                return pyobj;
        }
}

std::vector<py::dict> processRleToBoundary(const std::vector<RLE>& rles,
                                           const double& dilation_ratio,
                                           const size_t& cpu_count) {
        py::gil_scoped_release release;
        std::vector<std::tuple<uint64_t, uint64_t, std::string>> result(
            rles.size());

        auto process = [&rles, &result](size_t s, size_t e, double d) {
                for (size_t i = s; i < e; ++i) {
                        result[i] = rles[i].toBoundary(d).toTuple();
                }
        };

        size_t start = 0;
        // Adaptive step size based on input size and CPU count
        // For small batches, distribute work more evenly across threads
        size_t min_step = std::max(
            size_t(1),
            rles.size() / (cpu_count * 4));  // At least 4 work units per thread
        size_t max_step = 1000;              // Keep reasonable upper bound
        size_t step =
            std::max(min_step, std::min(max_step, rles.size() / cpu_count));
        size_t end = step;
        if (end > rles.size()) end = rles.size();

        while (start < rles.size()) {
                std::vector<std::future<void>> rle_futures;
                rle_futures.reserve(cpu_count);

                size_t thread = 0;
                try {
                        // Launch async tasks
                        for (thread = 0; thread < cpu_count; thread++) {
                                rle_futures.emplace_back(
                                    std::async(std::launch::async, process,
                                               start, end, dilation_ratio));

                                start += step;
                                end += step;

                                if (end > rles.size()) end = rles.size();
                                if (start >= rles.size()) {
                                        thread++;
                                        break;
                                }
                        }

                        // Wait for all tasks with proper exception handling
                        std::exception_ptr first_exception = nullptr;
                        for (size_t i = 0; i < rle_futures.size(); i++) {
                                try {
                                        rle_futures[i]
                                            .get();  // get() to catch
                                                     // exceptions from threads
                                } catch (...) {
                                        if (!first_exception) {
                                                first_exception =
                                                    std::current_exception();
                                        }
                                        // Continue waiting for other threads to
                                        // complete
                                }
                        }

                        // Re-throw the first exception if any occurred
                        if (first_exception) {
                                std::rethrow_exception(first_exception);
                        }

                } catch (...) {
                        // Ensure all futures are properly cleaned up on
                        // exception
                        for (auto& future : rle_futures) {
                                if (future.valid()) {
                                        try {
                                                future.wait();  // Ensure thread
                                                                // completion
                                        } catch (...) {
                                                // Ignore exceptions during
                                                // cleanup
                                        }
                                }
                        }
                        throw;  // Re-throw the original exception
                }
        }

        py::gil_scoped_acquire acquire;

        std::vector<py::dict> py_result(result.size());
        for (size_t i = 0; i < result.size(); i++) {
                py_result[i] = py::dict(
                    "size"_a = std::vector<uint64_t>{std::get<0>(result[i]),
                                                     std::get<1>(result[i])},
                    "counts"_a = py::bytes(std::get<2>(result[i])));
        }
        return py_result;
}

// Calculates and adds RLE masks (and optionally boundary RLEs) for all
// annotation objects in the input list. For each annotation with a
// "segmentation" field, computes the RLE mask using its associated image size.
// If boundary computation is enabled, also computes the boundary RLE for each
// annotation in parallel. The function updates each annotation in place by
// adding "rle" and, if requested, "boundary" fields. Parameters:
//   - anns: Vector of py::dict annotation objects, each representing an
//   instance annotation.
//   - image_info: Map from image_id to a tuple (height, width) for each image.
//   - compute_rle: If true, computes and assigns the RLE mask to each
//   annotation.
//   - compute_boundary: If true, computes and assigns the boundary RLE to each
//   annotation.
//   - dilation_ratio: Ratio of the mask diagonal for the erosion radius in
//   boundary extraction.
//   - cpu_count: Number of CPU threads to use for parallel boundary
//   computation.
// Returns:
//   - None. All updates are performed in place on the anns vector.
void calculateRleForAllAnnotations(
    const std::vector<py::dict>& anns,
    const std::unordered_map<uint64_t, std::tuple<uint64_t, uint64_t>>&
        image_info,
    const bool& compute_rle, const bool& compute_boundary,
    const double& dilation_ratio, const size_t& cpu_count) {
        if (!compute_rle) return;

        size_t ann_count = anns.size();

        // Adaptive batch size to prevent excessive memory usage
        const size_t MAX_BATCH_SIZE =
            2000;  // Adjustable based on available memory
        const size_t MIN_BATCH_SIZE = 100;
        size_t batch_size =
            std::min(MAX_BATCH_SIZE, std::max(MIN_BATCH_SIZE, ann_count / 10));

        // Process annotations in batches to reduce memory footprint
        for (size_t batch_start = 0; batch_start < ann_count;
             batch_start += batch_size) {
                size_t batch_end =
                    std::min(batch_start + batch_size, ann_count);
                size_t current_batch_size = batch_end - batch_start;

                // Create RLE objects for current batch
                std::vector<RLE> batch_rles;
                batch_rles.reserve(current_batch_size);
                std::vector<size_t> valid_indices;
                valid_indices.reserve(current_batch_size);

                for (size_t i = batch_start; i < batch_end; i++) {
                        if (anns[i].contains("segmentation")) {
                                uint64_t image_id =
                                    anns[i]["image_id"].cast<uint64_t>();
                                auto image_hw = image_info.at(image_id);
                                batch_rles.emplace_back(RLE::frSegm(
                                    anns[i]["segmentation"],
                                    std::get<1>(image_hw),  // width
                                    std::get<0>(image_hw)   // height
                                    ));
                                valid_indices.push_back(i);
                        }
                }

                // Process boundaries for current batch if needed
                std::vector<py::dict> batch_boundaries;
                if (compute_boundary && !batch_rles.empty()) {
                        batch_boundaries = processRleToBoundary(
                            batch_rles, dilation_ratio, cpu_count);
                }

                // Store results back to annotations
                for (size_t j = 0; j < valid_indices.size(); j++) {
                        size_t ann_idx = valid_indices[j];
                        anns[ann_idx]["rle"] = batch_rles[j].toDict();
                        if (compute_boundary && j < batch_boundaries.size()) {
                                anns[ann_idx]["boundary"] = batch_boundaries[j];
                        }
                }
        }
}

}  // namespace Mask

}  // namespace mask_api
