// Copyright (c) Facebook, Inc. and its affiliates.
#include "cocoeval.h"
#include <time.h>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <future>
using namespace pybind11::literals;

namespace coco_eval
{

  template <typename T>
  std::vector<T> list_to_vec(const py::list &l)
  {
    const size_t n = py::len(l);
    std::vector<T> v(n);
    for (size_t i = 0; i < n; ++i)
    {
      v[i] = l[i].cast<T>();
    }
    return v;
  }

  namespace COCOeval
  {

    // Returns the current local time as a string in the format "YYYY-MM-DD HH:MM:SS"
    std::string get_current_local_time_string()
    {
      time_t rawtime;
      struct tm local_time;
      char buffer[200];
      time(&rawtime);

#ifdef _WIN32
      localtime_s(&local_time, &rawtime);
#else
      localtime_r(&rawtime, &local_time);
#endif
      // Corrected format: "%Y-%m-%d %H:%M:%S" (minutes were missing in original)
      strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &local_time);
      return std::string(buffer);
    }

    // Sorts detections from highest to lowest score using stable_sort to match the original COCO API behavior.
    // Arguments:
    //   detection_instances: Vector of detection instances to be sorted.
    //   detection_sorted_indices: Output vector of indices sorted by detection score in descending order.
    void SortInstancesByDetectionScore(
        const std::vector<InstanceAnnotation> &detection_instances,
        std::vector<uint64_t> *detection_sorted_indices)
    {
      // Resize output vector to match input size
      detection_sorted_indices->resize(detection_instances.size());
      // Fill indices [0, 1, ..., n-1]
      std::iota(detection_sorted_indices->begin(), detection_sorted_indices->end(), 0);

      // Sort indices by score in descending order, using stable_sort for stable ordering of equal scores
      std::stable_sort(
          detection_sorted_indices->begin(),
          detection_sorted_indices->end(),
          [&detection_instances](uint64_t j1, uint64_t j2)
          {
            return detection_instances[j1].score > detection_instances[j2].score;
          });
    }

    // Partitions ground truth objects based on whether they should be ignored according to area constraints.
    // Sorts indices so that non-ignored instances come before ignored ones, preserving the original order within each group.
    // Arguments:
    //   area_range: Minimum and maximum area thresholds [min_area, max_area].
    //   ground_truth_instances: Vector of ground truth objects.
    //   ground_truth_sorted_indices: Output vector of indices, sorted so that non-ignored objects appear first.
    //   ignores: Output vector of bools indicating whether each instance should be ignored.
    void SortInstancesByIgnore(
        const std::array<double, 2> &area_range,
        const std::vector<InstanceAnnotation> &ground_truth_instances,
        std::vector<uint64_t> *ground_truth_sorted_indices,
        std::vector<bool> *ignores)
    {
      // Clear and reserve space for ignores vector for efficiency
      ignores->clear();
      ignores->reserve(ground_truth_instances.size());

      // Mark objects to ignore if the 'ignore' flag is set, or area is out of range
      for (const auto &o : ground_truth_instances)
      {
        ignores->emplace_back(
            o.ignore || o.area < area_range[0] || o.area > area_range[1]);
      }

      // Initialize sorted indices [0, 1, ..., n-1]
      ground_truth_sorted_indices->resize(ground_truth_instances.size());
      std::iota(
          ground_truth_sorted_indices->begin(),
          ground_truth_sorted_indices->end(),
          0);

      // Sort indices so that non-ignored objects come before ignored ones
      std::stable_sort(
          ground_truth_sorted_indices->begin(),
          ground_truth_sorted_indices->end(),
          [ignores](uint64_t j1, uint64_t j2)
          {
            // Non-ignored (false) comes before ignored (true)
            return static_cast<int>((*ignores)[j1]) < static_cast<int>((*ignores)[j2]);
          });
    }

    // For each IOU threshold, greedily match each detected instance to a ground truth instance (if possible)
    // and store the results in the provided ImageEvaluation struct.
    // Arguments:
    //   detection_instances: Vector of detected object annotations.
    //   detection_sorted_indices: Indices of detections sorted by score (descending).
    //   ground_truth_instances: Vector of ground truth object annotations.
    //   ground_truth_sorted_indices: Indices of ground truths sorted by ignore status.
    //   ignores: Vector<bool> indicating which ground truth objects should be ignored.
    //   ious: Matrix of IOU values between each detection and ground truth.
    //   iou_thresholds: List of IOU thresholds for evaluation.
    //   area_range: Area range [min, max] for valid detections.
    //   results: Output struct to store results, including matches, ignores, and scores.
    void MatchDetectionsToGroundTruth(
        const std::vector<InstanceAnnotation> &detection_instances,
        const std::vector<uint64_t> &detection_sorted_indices,
        const std::vector<InstanceAnnotation> &ground_truth_instances,
        const std::vector<uint64_t> &ground_truth_sorted_indices,
        const std::vector<bool> &ignores,
        const std::vector<std::vector<double>> &ious,
        const std::vector<double> &iou_thresholds,
        const std::array<double, 2> &area_range,
        ImageEvaluation *results)
    {
      // Clear any previously stored matched annotations to avoid duplications
      results->matched_annotations.clear();

      const int num_iou_thresholds = static_cast<int>(iou_thresholds.size());
      const int num_ground_truth = static_cast<int>(ground_truth_sorted_indices.size());
      const int num_detections = static_cast<int>(detection_sorted_indices.size());

      // Prepare output buffers
      std::vector<int64_t> &ground_truth_matches = results->ground_truth_matches;
      ground_truth_matches.assign(num_iou_thresholds * num_ground_truth, 0);

      std::vector<int64_t> &detection_matches = results->detection_matches;
      detection_matches.assign(num_iou_thresholds * num_detections, 0);

      std::vector<bool> &detection_ignores = results->detection_ignores;
      detection_ignores.assign(num_iou_thresholds * num_detections, false);

      std::vector<bool> &ground_truth_ignores = results->ground_truth_ignores;
      ground_truth_ignores.resize(num_ground_truth);
      for (int g = 0; g < num_ground_truth; ++g)
      {
        ground_truth_ignores[g] = ignores[ground_truth_sorted_indices[g]];
      }

      // Main matching loop: for each IOU threshold, process all detections
      for (int t = 0; t < num_iou_thresholds; ++t)
      {
        double threshold = std::min(iou_thresholds[t], 1.0 - 1e-10);

        for (int d = 0; d < num_detections; ++d)
        {
          double best_iou = threshold;
          int match = -1;

          // Greedily find best ground truth match for this detection
          for (int g = 0; g < num_ground_truth; ++g)
          {
            // If this ground truth is matched and not a crowd, skip it
            if (ground_truth_matches[t * num_ground_truth + g] > 0 &&
                !ground_truth_instances[ground_truth_sorted_indices[g]].is_crowd)
            {
              continue;
            }

            // Optimization: can break early if we hit the first ignored GT after a non-ignored match
            if (match >= 0 && !ground_truth_ignores[match] && ground_truth_ignores[g])
            {
              break;
            }

            // Update best match if IOU is above threshold and is the best so far
            double iou = ious[d][ground_truth_sorted_indices[g]];
            if (iou >= best_iou)
            {
              best_iou = iou;
              match = g;
            }
          }

          // Store match results if there was a match
          if (match >= 0)
          {
            detection_ignores[t * num_detections + d] = ground_truth_ignores[match];
            detection_matches[t * num_detections + d] =
                ground_truth_instances[ground_truth_sorted_indices[match]].id;
            ground_truth_matches[t * num_ground_truth + match] =
                detection_instances[detection_sorted_indices[d]].id;

            results->matched_annotations.emplace_back(
                ground_truth_matches[t * num_ground_truth + match], // DT_ID
                detection_matches[t * num_detections + d],          // GT_ID
                best_iou);
          }

          // Set unmatched detections outside area range (or marked as lvis) to ignore
          const InstanceAnnotation &detection =
              detection_instances[detection_sorted_indices[d]];
          if (detection_matches[t * num_detections + d] == 0 &&
              (detection.area < area_range[0] || detection.area > area_range[1] || detection.lvis_mark))
          {
            detection_ignores[t * num_detections + d] = true;
          }
        }
      }

      // Store detection scores in sorted order
      results->detection_scores.resize(detection_sorted_indices.size());
      for (size_t d = 0; d < detection_sorted_indices.size(); ++d)
      {
        results->detection_scores[d] =
            detection_instances[detection_sorted_indices[d]].score;
      }
    }

    // Evaluates detection results for multiple images, categories, and area ranges.
    // For each combination of image, category, and area range, matches detections to ground truths
    // using IOU thresholds and stores evaluation results.
    //
    // Arguments:
    //   area_ranges:        Vector of [min, max] area thresholds.
    //   max_detections:     Maximum number of detections to evaluate per image/category (for top-scoring).
    //   iou_thresholds:     List of IOU thresholds.
    //   image_category_ious: 3D structure: [image][category][iou matrix].
    //   image_category_ground_truth_instances: 3D structure: [image][category][ground truth instances].
    //   image_category_detection_instances:    3D structure: [image][category][detection instances].
    //
    // Returns:
    //   results_all: Vector of ImageEvaluation objects for each image/category/area_range combination.
    std::vector<ImageEvaluation> EvaluateImages(
        const std::vector<std::array<double, 2>> &area_ranges,
        int max_detections,
        const std::vector<double> &iou_thresholds,
        const ImageCategoryInstances<std::vector<double>> &image_category_ious,
        const ImageCategoryInstances<InstanceAnnotation> &image_category_ground_truth_instances,
        const ImageCategoryInstances<InstanceAnnotation> &image_category_detection_instances)
    {
      const size_t num_area_ranges = area_ranges.size();
      const size_t num_images = image_category_ground_truth_instances.size();
      const size_t num_categories =
          (image_category_ious.size() > 0) ? image_category_ious[0].size() : 0;

      std::vector<uint64_t> detection_sorted_indices;
      std::vector<uint64_t> ground_truth_sorted_indices;
      std::vector<bool> ignores;

      // Preallocate result vector.
      // Index mapping: [category * num_area_ranges * num_images + area * num_images + image]
      std::vector<ImageEvaluation> results_all(num_images * num_area_ranges * num_categories);

      for (size_t i = 0; i < num_images; ++i)
      {
        for (size_t c = 0; c < num_categories; ++c)
        {
          const auto &ground_truth_instances = image_category_ground_truth_instances[i][c];
          const auto &detection_instances = image_category_detection_instances[i][c];

          // Sort detections by score (descending).
          SortInstancesByDetectionScore(detection_instances, &detection_sorted_indices);
          if (detection_sorted_indices.size() > static_cast<size_t>(max_detections))
          {
            detection_sorted_indices.resize(max_detections);
          }

          for (size_t a = 0; a < num_area_ranges; ++a)
          {
            // Partition ground truth objects by ignore criteria.
            SortInstancesByIgnore(
                area_ranges[a], ground_truth_instances,
                &ground_truth_sorted_indices, &ignores);

            // Greedily match detections to ground truth per IOU threshold.
            MatchDetectionsToGroundTruth(
                detection_instances,
                detection_sorted_indices,
                ground_truth_instances,
                ground_truth_sorted_indices,
                ignores,
                image_category_ious[i][c],
                iou_thresholds,
                area_ranges[a],
                &results_all[c * num_area_ranges * num_images + a * num_images + i]);
          }
        }
      }
      return results_all;
    }

    // Helper function to accumulate detection results across images for a specific category, area range, and max_detections setting.
    // Extracts and sorts all applicable detections, returning the number of valid (non-ignored) ground truth objects.
    // Arguments:
    //   evaluations:                Vector of ImageEvaluation objects.
    //   evaluation_index:           Starting index for the relevant group in evaluations (for the given category/area/max_det).
    //   num_images:                 Number of images in the group.
    //   max_detections:             Maximum detections per image to consider.
    //   evaluation_indices:         Output: indices into evaluations[] for each detection instance.
    //   detection_scores:           Output: detection score for each instance.
    //   detection_sorted_indices:   Output: sorted indices (by score, descending) for the detections.
    //   image_detection_indices:    Output: index of the detection within its image.
    // Returns: number of valid (non-ignored) ground truth objects.
    int BuildSortedDetectionList(
        const std::vector<ImageEvaluation> &evaluations,
        const int64_t evaluation_index,
        const int64_t num_images,
        const int max_detections,
        std::vector<uint64_t> *evaluation_indices,
        std::vector<double> *detection_scores,
        std::vector<uint64_t> *detection_sorted_indices,
        std::vector<uint64_t> *image_detection_indices)
    {
      assert(evaluations.size() >= static_cast<size_t>(evaluation_index + num_images));

      // Prepare output containers
      image_detection_indices->clear();
      evaluation_indices->clear();
      detection_scores->clear();
      image_detection_indices->reserve(num_images * max_detections);
      evaluation_indices->reserve(num_images * max_detections);
      detection_scores->reserve(num_images * max_detections);

      int num_valid_ground_truth = 0;

      for (int64_t i = 0; i < num_images; ++i)
      {
        const ImageEvaluation &evaluation = evaluations[evaluation_index + i];

        // Collect up to max_detections detection scores per image
        for (int d = 0;
             d < static_cast<int>(evaluation.detection_scores.size()) && d < max_detections;
             ++d)
        {
          evaluation_indices->emplace_back(evaluation_index + i);
          image_detection_indices->emplace_back(d);
          detection_scores->emplace_back(evaluation.detection_scores[d]);
        }

        // Count valid (non-ignored) ground truth instances
        for (bool ground_truth_ignore : evaluation.ground_truth_ignores)
        {
          if (!ground_truth_ignore)
          {
            ++num_valid_ground_truth;
          }
        }
      }

      // Sort detections by decreasing score (stable sort for reproducibility)
      detection_sorted_indices->resize(detection_scores->size());
      std::iota(detection_sorted_indices->begin(), detection_sorted_indices->end(), 0);
      std::stable_sort(
          detection_sorted_indices->begin(),
          detection_sorted_indices->end(),
          [&detection_scores](size_t j1, size_t j2)
          {
            return (*detection_scores)[j1] > (*detection_scores)[j2];
          });

      return num_valid_ground_truth;
    }

    // Helper function for Accumulate()
    // Computes a precision-recall curve given a sorted list of detected instances.
    // See BuildSortedDetectionList() for inputs.
    // Arguments:
    //   precisions_out_index:    Index for output precision vector
    //   precisions_out_stride:   Stride for output precision/scores vector
    //   recalls_out_index:       Index for output recall vector
    //   recall_thresholds:       List of recall thresholds to sample at
    //   iou_threshold_index:     Index of the IOU threshold being evaluated
    //   num_iou_thresholds:      Total number of IOU thresholds
    //   num_valid_ground_truth:  Number of valid (non-ignored) ground truth instances
    //   evaluations:             List of ImageEvaluation (per-image detection/gt info)
    //   evaluation_indices:      Indices into evaluations for each detection
    //   detection_scores:        Scores for each detection
    //   detection_sorted_indices:Sorted permutation of detection indices (by score)
    //   image_detection_indices: Index of detection within its image
    //   precisions, recalls:     Temporary storage for per-instance precision/recall
    //   precisions_out, scores_out, recalls_out: Output buffers for full curves
    void ComputePrecisionRecallCurve(
        const int64_t precisions_out_index,
        const int64_t precisions_out_stride,
        const int64_t recalls_out_index,
        const std::vector<double> &recall_thresholds,
        const int iou_threshold_index,
        const int num_iou_thresholds,
        const int num_valid_ground_truth,
        const std::vector<ImageEvaluation> &evaluations,
        const std::vector<uint64_t> &evaluation_indices,
        const std::vector<double> &detection_scores,
        const std::vector<uint64_t> &detection_sorted_indices,
        const std::vector<uint64_t> &image_detection_indices,
        std::vector<double> *precisions,
        std::vector<double> *recalls,
        std::vector<double> *precisions_out,
        std::vector<double> *scores_out,
        std::vector<double> *recalls_out)
    {
      assert(recalls_out->size() > static_cast<size_t>(recalls_out_index));

      // Clear and reserve temporary output vectors
      int64_t true_positives_sum = 0, false_positives_sum = 0;
      precisions->clear();
      recalls->clear();
      precisions->reserve(detection_sorted_indices.size());
      recalls->reserve(detection_sorted_indices.size());
      assert(!evaluations.empty() || detection_sorted_indices.empty());

      // Compute precision/recall for each detection in sorted order
      for (auto detection_sorted_index : detection_sorted_indices)
      {
        const ImageEvaluation &evaluation =
            evaluations[evaluation_indices[detection_sorted_index]];
        const auto num_detections =
            evaluation.detection_matches.size() / num_iou_thresholds;
        const auto detection_index = iou_threshold_index * num_detections +
                                     image_detection_indices[detection_sorted_index];
        assert(evaluation.detection_matches.size() > detection_index);
        assert(evaluation.detection_ignores.size() > detection_index);
        const int64_t detection_match =
            evaluation.detection_matches[detection_index];
        const bool detection_ignores =
            evaluation.detection_ignores[detection_index];
        const auto true_positive = detection_match > 0 && !detection_ignores;
        const auto false_positive = detection_match == 0 && !detection_ignores;
        if (true_positive)
        {
          ++true_positives_sum;
        }
        if (false_positive)
        {
          ++false_positives_sum;
        }

        const double recall =
            num_valid_ground_truth > 0 ? static_cast<double>(true_positives_sum) / num_valid_ground_truth : 0.0;
        recalls->emplace_back(recall);
        const int64_t num_valid_detections = true_positives_sum + false_positives_sum;
        const double precision = num_valid_detections > 0
                                     ? static_cast<double>(true_positives_sum) / num_valid_detections
                                     : 0.0;
        precisions->emplace_back(precision);
      }

      (*recalls_out)[recalls_out_index] = !recalls->empty() ? recalls->back() : 0;

      // Make precision non-increasing (interpolated)
      for (int64_t i = static_cast<int64_t>(precisions->size()) - 1; i > 0; --i)
      {
        if ((*precisions)[i] > (*precisions)[i - 1])
        {
          (*precisions)[i - 1] = (*precisions)[i];
        }
      }

      // Sample the precision/recall lists at each recall threshold
      for (size_t r = 0; r < recall_thresholds.size(); ++r)
      {
        // First index in recalls >= recall_thresholds[r]
        auto low = std::lower_bound(recalls->begin(), recalls->end(), recall_thresholds[r]);
        size_t precisions_index = static_cast<size_t>(low - recalls->begin());

        const auto results_ind = precisions_out_index + r * precisions_out_stride;
        assert(results_ind < precisions_out->size());
        assert(results_ind < scores_out->size());
        if (precisions_index < precisions->size())
        {
          (*precisions_out)[results_ind] = (*precisions)[precisions_index];
          (*scores_out)[results_ind] =
              detection_scores[detection_sorted_indices[precisions_index]];
        }
        else
        {
          (*precisions_out)[results_ind] = 0;
          (*scores_out)[results_ind] = 0;
        }
      }
    }

    py::dict Accumulate(
        const py::object &params,
        const std::vector<ImageEvaluation> &evaluations)
    {
      // Convert Python lists to C++ vectors for efficiency
      const auto recall_thresholds = list_to_vec<double>(params.attr("recThrs"));
      const auto max_detections = list_to_vec<int>(params.attr("maxDets"));

      // Cache Python attribute accesses and casts
      const auto iouThrs = params.attr("iouThrs");
      const auto recThrs = params.attr("recThrs");
      const auto useCats = params.attr("useCats").cast<int>();
      const auto catIds = params.attr("catIds");
      const auto areaRng = params.attr("areaRng");
      const auto imgIds = params.attr("imgIds");

      const int num_iou_thresholds = static_cast<int>(py::len(iouThrs));
      const int num_recall_thresholds = static_cast<int>(py::len(recThrs));
      const int num_categories = useCats == 1 ? static_cast<int>(py::len(catIds)) : 1;
      const int num_area_ranges = static_cast<int>(py::len(areaRng));
      const int num_max_detections = static_cast<int>(max_detections.size());
      const int num_images = static_cast<int>(py::len(imgIds));

      // Pre-allocate output arrays with -1 as default value
      std::vector<double> precisions_out(
          static_cast<size_t>(num_iou_thresholds) * num_recall_thresholds * num_categories *
              num_area_ranges * num_max_detections,
          -1.0);
      std::vector<double> recalls_out(
          static_cast<size_t>(num_iou_thresholds) * num_categories * num_area_ranges *
              num_max_detections,
          -1.0);
      std::vector<double> scores_out(
          static_cast<size_t>(num_iou_thresholds) * num_recall_thresholds * num_categories *
              num_area_ranges * num_max_detections,
          -1.0);

      // Reserve memory for large vectors to avoid reallocations
      std::vector<uint64_t> evaluation_indices;
      std::vector<double> detection_scores;
      std::vector<uint64_t> detection_sorted_indices;
      std::vector<uint64_t> image_detection_indices;
      std::vector<double> precisions, recalls;

      // Main nested loops: optimize loop ordering for cache locality if possible
      for (int c = 0; c < num_categories; ++c)
      {
        for (int a = 0; a < num_area_ranges; ++a)
        {
          for (int m = 0; m < num_max_detections; ++m)
          {
            // Index for the flattened evaluations[] list
            const int64_t evaluations_index =
                static_cast<int64_t>(c) * num_area_ranges * num_images + a * num_images;

            // Reset vectors in-place to avoid reallocations, keep capacity
            evaluation_indices.clear();
            detection_scores.clear();
            detection_sorted_indices.clear();
            image_detection_indices.clear();

            int num_valid_ground_truth = BuildSortedDetectionList(
                evaluations,
                evaluations_index,
                num_images,
                max_detections[m],
                &evaluation_indices,
                &detection_scores,
                &detection_sorted_indices,
                &image_detection_indices);

            if (num_valid_ground_truth == 0)
            {
              continue;
            }

            for (int t = 0; t < num_iou_thresholds; ++t)
            {
              // Calculate recall output index (flattened multi-dimensional array)
              const int64_t recalls_out_index =
                  static_cast<int64_t>(t) * num_categories * num_area_ranges * num_max_detections +
                  c * num_area_ranges * num_max_detections +
                  a * num_max_detections + m;

              // Calculate precisions/scores output index and stride
              const int64_t precisions_out_stride =
                  static_cast<int64_t>(num_categories) * num_area_ranges * num_max_detections;
              const int64_t precisions_out_index =
                  static_cast<int64_t>(t) * num_recall_thresholds *
                      num_categories * num_area_ranges * num_max_detections +
                  c * num_area_ranges * num_max_detections +
                  a * num_max_detections + m;

              ComputePrecisionRecallCurve(
                  precisions_out_index,
                  precisions_out_stride,
                  recalls_out_index,
                  recall_thresholds,
                  t,
                  num_iou_thresholds,
                  num_valid_ground_truth,
                  evaluations,
                  evaluation_indices,
                  detection_scores,
                  detection_sorted_indices,
                  image_detection_indices,
                  &precisions,
                  &recalls,
                  &precisions_out,
                  &scores_out,
                  &recalls_out);
            }
          }
        }
      }

      const int evaluations_size = static_cast<int>(evaluations.size());

      // Use unordered_map for fast lookups of matched annotations
      std::unordered_map<std::string, double> matched;
      matched.reserve(evaluations.size() * 2); // Reserve assuming 2 matches per evaluation on average

      // Fill matched map with maximum IoU for each (dt_id, gt_id) pair
      for (const auto &eval : evaluations)
      {
        for (const auto &matched_annotation : eval.matched_annotations)
        {
          std::string key = std::to_string(matched_annotation.dt_id) + "_" + std::to_string(matched_annotation.gt_id);

          auto it = matched.find(key);
          if (it != matched.end())
          {
            if (it->second < matched_annotation.iou)
            {
              it->second = matched_annotation.iou;
            }
          }
          else
          {
            matched.emplace(std::move(key), matched_annotation.iou);
          }
        }
      }

      // Prepare output array shapes
      std::vector<int64_t> counts = {
          num_iou_thresholds,
          num_recall_thresholds,
          num_categories,
          num_area_ranges,
          num_max_detections};

      std::vector<int64_t> recall_counts = {
          num_iou_thresholds,
          num_categories,
          num_area_ranges,
          num_max_detections};

      // Return results in a Python dictionary
      return py::dict(
          "params"_a = params,
          "counts"_a = counts,
          "date"_a = py::str(get_current_local_time_string()),

          "matched"_a = matched,

          // Precision and scores: shape [T, R, K, A, M]
          "precision"_a = py::array(precisions_out.size(), precisions_out.data()).reshape(counts),
          "scores"_a = py::array(scores_out.size(), scores_out.data()).reshape(counts),

          // Recall: shape [T, K, A, M]
          "recall"_a = py::array(recalls_out.size(), recalls_out.data()).reshape(recall_counts),
          "evaluations_size"_a = evaluations_size);
    }

    // EvaluateAccumulate computes evaluation results for all images, categories, and area ranges,
    // and then accumulates the statistics using Accumulate.
    // Arguments:
    //   params - Python object with evaluation parameters
    //   image_category_ious - IoU values for each image/category pair
    //   image_category_ground_truth_instances - ground truth instances for each image/category pair
    //   image_category_detection_instances - detection instances for each image/category pair
    py::dict EvaluateAccumulate(
        const py::object &params,
        const ImageCategoryInstances<std::vector<double>> &image_category_ious,
        const ImageCategoryInstances<InstanceAnnotation> &image_category_ground_truth_instances,
        const ImageCategoryInstances<InstanceAnnotation> &image_category_detection_instances)
    {
      // Convert Python objects to C++ vectors for efficiency and type safety
      const std::vector<int> max_detections = list_to_vec<int>(params.attr("maxDets"));
      const std::vector<std::array<double, 2>> area_ranges = list_to_vec<std::array<double, 2>>(params.attr("areaRng"));
      const std::vector<double> iou_thresholds = list_to_vec<double>(params.attr("iouThrs"));

      // Evaluate all images/categories/ranges with the largest maxDet value
      std::vector<ImageEvaluation> result = EvaluateImages(
          area_ranges,
          max_detections.back(), // Use the largest maxDet for thorough evaluation
          iou_thresholds,
          image_category_ious,
          image_category_ground_truth_instances,
          image_category_detection_instances);

      // Accumulate the precision/recall and other stats for reporting
      return Accumulate(params, result);
    }

    // Appends an annotation to the dataset for a specific (img_id, cat_id) pair.
    // Uses emplace_back for efficient insertion. Accepts ann as const reference to avoid unnecessary copying.
    void Dataset::append(int64_t img_id, int64_t cat_id, const py::dict &ann)
    {
      // Use emplace_back to construct py::dict in-place for efficiency.
      data[{img_id, cat_id}].emplace_back(ann);
    }

    // Removes all stored annotations and frees internal memory used by the data container.
    void Dataset::clean()
    {
      data.clear();
      // Optionally reclaim memory by swapping with an empty map (C++17 idiom)
      std::unordered_map<std::pair<int64_t, int64_t>, std::vector<py::dict>, hash_pair>().swap(data);
    }

    // Serializes the Dataset into a Python tuple for pickling support.
    // Stores the keys and values as two separate vectors for efficient transfer to Python.
    py::tuple Dataset::make_tuple() const
    {
      // Preallocate storage for better memory efficiency.
      std::vector<std::pair<int64_t, int64_t>> keys;
      std::vector<std::vector<py::dict>> values;
      keys.reserve(data.size());
      values.reserve(data.size());

      // Iterate over all key-value pairs and collect them.
      for (const auto &kv : data)
      {
        keys.push_back(kv.first);
        values.push_back(kv.second);
      }
      // Return as Python tuple (keys, values).
      return py::make_tuple(keys, values);
    }

    // Loads the Dataset state from a Python tuple (typically for unpickling).
    // The tuple must contain two elements: a vector of keys and a vector of value vectors.
    // Throws a runtime error if the state is invalid.
    void Dataset::load_tuple(py::tuple pickle_data)
    {
      if (pickle_data.size() != 2)
        throw std::runtime_error("Invalid state! Tuple must have 2 elements.");

      // Cast Python objects to C++ vectors.
      std::vector<std::pair<int64_t, int64_t>> keys = pickle_data[0].cast<std::vector<std::pair<int64_t, int64_t>>>();
      std::vector<std::vector<py::dict>> values = pickle_data[1].cast<std::vector<std::vector<py::dict>>>();

      if (keys.size() != values.size())
        throw std::runtime_error("Invalid state! Keys and values vectors must have the same size.");

      // Clear existing data and reserve memory for efficiency.
      data.clear();
      data.reserve(keys.size());

      // Insert each key-value pair using move semantics for optimal performance.
      for (size_t i = 0; i < keys.size(); ++i)
      {
        data.emplace(std::move(keys[i]), std::move(values[i]));
      }
    }

    // Returns a vector of Python dictionaries for a given (img_id, cat_id) pair.
    // If the pair is not found, returns an empty vector.
    // Uses find() to avoid unnecessary construction or lookup.
    std::vector<py::dict> Dataset::get(const int64_t &img_id, const int64_t &cat_id)
    {
      const std::pair<int64_t, int64_t> key(img_id, cat_id);

      auto it = data.find(key);
      if (it != data.end())
      {
        return it->second;
      }
      else
      {
        // Return empty vector if key is not found.
        return {};
      }
    }

    // Parses a py::dict annotation into an InstanceAnnotation object.
    // Extracts fields by name and casts to appropriate type.
    // Optimized for C++17: uses const references, avoids unnecessary string copies.
    InstanceAnnotation parseInstanceAnnotation(const py::dict &ann)
    {
      uint64_t id = 0;
      double score = 0.0;
      double area = 0.0;
      bool is_crowd = false;
      bool ignore = false;
      bool lvis_mark = false;

      // Iterate using const reference for efficiency.
      for (const auto &item : ann)
      {
        const auto &key_obj = item.first;
        const auto &val_obj = item.second;

        // Use string_view for faster string comparison in C++17
        std::string_view key = key_obj.cast<std::string>();

        if (key == "id")
        {
          id = val_obj.cast<uint64_t>();
        }
        else if (key == "score")
        {
          score = val_obj.cast<double>();
        }
        else if (key == "area")
        {
          area = val_obj.cast<double>();
        }
        else if (key == "is_crowd" || key == "iscrowd")
        {
          is_crowd = val_obj.cast<bool>();
        }
        else if (key == "ignore")
        {
          ignore = val_obj.cast<bool>();
        }
        else if (key == "lvis_mark")
        {
          lvis_mark = val_obj.cast<bool>();
        }
      }
      // Construct and return the annotation.
      return InstanceAnnotation(id, score, area, is_crowd, ignore, lvis_mark);
    }

    // Returns a vector of InstanceAnnotation objects for a given (img_id, cat_id) pair.
    // Uses reserve() for performance and emplace_back for efficient insertion.
    std::vector<InstanceAnnotation> Dataset::get_cpp_annotations(
        const int64_t &img_id, const int64_t &cat_id)
    {
      std::vector<py::dict> anns = get(img_id, cat_id);
      std::vector<InstanceAnnotation> result;
      result.reserve(anns.size()); // Reserve space to avoid reallocations.

      // Convert each py::dict annotation to InstanceAnnotation.
      for (const auto &ann : anns)
      {
        result.emplace_back(parseInstanceAnnotation(ann));
      }
      return result;
    }

    // Returns all InstanceAnnotations for each combination of img_ids and cat_ids.
    // If useCats is false, all category results for an image are merged into a single vector.
    // Optimized for better memory management and clarity.
    std::vector<std::vector<std::vector<InstanceAnnotation>>> Dataset::get_cpp_instances(
        const std::vector<int64_t> &img_ids,
        const std::vector<int64_t> &cat_ids,
        const bool &useCats)
    {
      std::vector<std::vector<std::vector<InstanceAnnotation>>> result;
      result.reserve(img_ids.size()); // Reserve space for image indices

      for (size_t i = 0; i < img_ids.size(); ++i)
      {
        int64_t img_id = img_ids[i];

        if (useCats)
        {
          std::vector<std::vector<InstanceAnnotation>> cat_results;
          cat_results.reserve(cat_ids.size()); // Reserve space for categories

          for (size_t j = 0; j < cat_ids.size(); ++j)
          {
            int64_t cat_id = cat_ids[j];
            cat_results.emplace_back(get_cpp_annotations(img_id, cat_id));
          }
          result.emplace_back(std::move(cat_results));
        }
        else
        {
          // Single vector to merge all categories for this image
          std::vector<InstanceAnnotation> merged;
          for (size_t j = 0; j < cat_ids.size(); ++j)
          {
            int64_t cat_id = cat_ids[j];
            std::vector<InstanceAnnotation> anns = get_cpp_annotations(img_id, cat_id);
            merged.insert(merged.end(),
                          std::make_move_iterator(anns.begin()),
                          std::make_move_iterator(anns.end()));
          }
          // Wrap merged vector in an outer vector for consistency
          result.emplace_back(1, std::move(merged));
        }
      }
      return result;
    }

    // Returns all py::dict annotations for each combination of img_ids and cat_ids.
    // If useCats is false, all category results for an image are merged into a single vector.
    // Optimized for better memory management and clarity.
    std::vector<std::vector<std::vector<py::dict>>> Dataset::get_instances(
        const std::vector<int64_t> &img_ids,
        const std::vector<int64_t> &cat_ids,
        const bool &useCats)
    {
      std::vector<std::vector<std::vector<py::dict>>> result;
      result.reserve(img_ids.size()); // Reserve space for images

      for (size_t i = 0; i < img_ids.size(); ++i)
      {
        int64_t img_id = img_ids[i];

        if (useCats)
        {
          std::vector<std::vector<py::dict>> cat_results;
          cat_results.reserve(cat_ids.size()); // Reserve space for categories

          for (size_t j = 0; j < cat_ids.size(); ++j)
          {
            int64_t cat_id = cat_ids[j];
            cat_results.emplace_back(get(img_id, cat_id));
          }
          result.emplace_back(std::move(cat_results));
        }
        else
        {
          // Single vector to merge all categories for this image
          std::vector<py::dict> merged;
          for (size_t j = 0; j < cat_ids.size(); ++j)
          {
            int64_t cat_id = cat_ids[j];
            std::vector<py::dict> anns = get(img_id, cat_id);
            merged.insert(merged.end(),
                          std::make_move_iterator(anns.begin()),
                          std::make_move_iterator(anns.end()));
          }
          // Wrap merged vector in an outer vector for consistency
          result.emplace_back(1, std::move(merged));
        }
      }
      return result;
    }

    // Computes the Area Under Curve (AUC) for precision-recall.
    // Uses the trapezoidal rule by accumulating area increments for each recall step with corresponding precision.
    // Ensures that precision is monotonically non-increasing.
    // Arguments:
    //   recall_list: vector of recall values (must be sorted in increasing order)
    //   precision_list: vector of precision values (same size as recall_list)
    long double calc_auc(const std::vector<long double> &recall_list, const std::vector<long double> &precision_list)
    {
      // Make a copy of precision_list to enforce monotonicity.
      std::vector<long double> mpre = precision_list;

      // Ensure precision is monotonically non-increasing, right to left.
      for (size_t i = mpre.size(); i-- > 1;) // i from size-1 down to 1
      {
        mpre[i - 1] = std::max(mpre[i - 1], mpre[i]);
      }

      long double result = 0;

      // Calculate area under the curve using the modified precision.
      for (size_t i = 1; i < recall_list.size(); ++i)
      {
        if (recall_list[i - 1] != recall_list[i])
        {
          result += (recall_list[i] - recall_list[i - 1]) * mpre[i];
        }
      }

      return result;
    }

  } // namespace COCOeval

} // namespace coco_eval
