// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

namespace py = pybind11;

namespace coco_eval
{

  namespace COCOeval
  {

    // Annotation data for a single object instance in an image
    struct InstanceAnnotation
    {
      InstanceAnnotation(
          uint64_t id,
          double score,
          double area,
          bool is_crowd,
          bool ignore,
          bool lvis_mark)
          : id{id}, score{score}, area{area}, is_crowd{is_crowd}, ignore{ignore}, lvis_mark{lvis_mark} {}
      uint64_t id;
      double score = 0.;
      double area = 0.;
      bool is_crowd = false;
      bool ignore = false;
      bool lvis_mark = false;
    };

    // Stores the match between a detected instance and a ground truth instance
    struct MatchedAnnotation
    {
      MatchedAnnotation(
          uint64_t dt_id,
          uint64_t gt_id,
          double iou) : dt_id{dt_id}, gt_id{gt_id}, iou{iou} {}
      uint64_t dt_id;
      uint64_t gt_id;
      double iou;
    };

    // Stores intermediate results for evaluating detection results for a single
    // image that has D detected instances and G ground truth instances. This stores
    // matches between detected and ground truth instances
    struct ImageEvaluation
    {
      // For each of the D detected instances, the id of the matched ground truth
      // instance, or 0 if unmatched
      std::vector<int64_t> detection_matches;

      std::vector<int64_t> ground_truth_matches;
      // The detection score of each of the D detected instances
      std::vector<double> detection_scores;

      // Marks whether or not each of G instances was ignored from evaluation (e.g.,
      // because it's outside area_range)
      std::vector<bool> ground_truth_ignores;

      // Marks whether or not each of D instances was ignored from evaluation (e.g.,
      // because it's outside aRng)
      std::vector<bool> detection_ignores;

      std::vector<MatchedAnnotation> matched_annotations;
    };

    template <class T>
    inline void hash_combine(std::size_t &seed, const T &v)
    {
      std::hash<T> hasher;
      seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    struct hash_pair
    {
      std::size_t operator()(const std::pair<int64_t, int64_t> &p) const
      {
        std::size_t h = 0;
        hash_combine(h, p.first);
        hash_combine(h, p.second);
        return h;
      }
    };

    class Dataset
    {
    public:
      Dataset()
      {
        this->data.reserve(8192); // RESERVING SPACE BEFOREHAND
        // this->data.max_load_factor(0.25); // DECREASING MAX_LOAD_FACTOR
      }
      void append(int64_t img_id, int64_t cat_id, py::dict ann);
      void clean();

      // pickle
      py::tuple make_tuple() const ;
      void load_tuple(py::tuple data);

      std::vector<py::dict> get(const int64_t &img_id, const int64_t &cat_id);
      std::vector<InstanceAnnotation> get_cpp_annotations(
          const int64_t &img_id, const int64_t &cat_id);

      std::vector<std::vector<std::vector<InstanceAnnotation>>> get_cpp_instances(
          const std::vector<int64_t> &img_ids, const std::vector<int64_t> &cat_ids, const bool &useCats);

      std::vector<std::vector<std::vector<py::dict>>> get_instances(
          const std::vector<int64_t> &img_ids, const std::vector<int64_t> &cat_ids, const bool &useCats);

    private:
      std::unordered_map<std::pair<int64_t, int64_t>, std::vector<py::dict>, hash_pair> data;
    };

    template <class T>
    using ImageCategoryInstances = std::vector<std::vector<std::vector<T>>>;

    // C++ implementation of COCO API cocoeval.py::COCOeval.evaluateImg().  For each
    // combination of image, category, area range settings, and IOU thresholds to
    // evaluate, it matches detected instances to ground truth instances and stores
    // the results into a vector of ImageEvaluation results, which will be
    // interpreted by the COCOeval::Accumulate() function to produce precion-recall
    // curves.  The parameters of nested vectors have the following semantics:
    //   image_category_ious[i][c][d][g] is the intersection over union of the d'th
    //     detected instance and g'th ground truth instance of
    //     category category_ids[c] in image image_ids[i]
    //   image_category_ground_truth_instances[i][c] is a vector of ground truth
    //     instances in image image_ids[i] of category category_ids[c]
    //   image_category_detection_instances[i][c] is a vector of detected
    //     instances in image image_ids[i] of category category_ids[c]
    std::vector<ImageEvaluation> EvaluateImages(
        const std::vector<std::array<double, 2>> &area_ranges, // vector of 2-tuples
        int max_detections,
        const std::vector<double> &iou_thresholds,
        const ImageCategoryInstances<std::vector<double>> &image_category_ious,
        const ImageCategoryInstances<InstanceAnnotation> &
            image_category_ground_truth_instances,
        const ImageCategoryInstances<InstanceAnnotation> &
            image_category_detection_instances);

    // C++ implementation of COCOeval.accumulate(), which generates precision
    // recall curves for each set of category, IOU threshold, detection area range,
    // and max number of detections parameters.  It is assumed that the parameter
    // evaluations is the return value of the functon COCOeval::EvaluateImages(),
    // which was called with the same parameter settings params
    py::dict Accumulate(
        const py::object &params,
        const std::vector<ImageEvaluation> &evalutations);

    py::dict EvaluateAccumulate(
        const py::object &params,
        const ImageCategoryInstances<std::vector<double>> &image_category_ious,
        const ImageCategoryInstances<InstanceAnnotation> &
            image_category_ground_truth_instances,
        const ImageCategoryInstances<InstanceAnnotation> &
            image_category_detection_instances);

    long double calc_auc(const std::vector<long double> &recall_list, const std::vector<long double> &precision_list);
    long double _summarize(const int &ap, const double &iouThr, const std::string &areaRng, const int &maxDet, const std::vector<int> &catIds, const py::object &params, const std::vector<size_t> &counts, const py::object &nums_array);
  } // namespace COCOeval
} // namespace coco_eval
