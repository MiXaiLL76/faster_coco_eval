#pragma once

#include <cstdint>
#include <vector>

namespace coco_eval {
namespace COCOeval {

// Annotation data for a single object instance in an image
struct InstanceAnnotation {
        InstanceAnnotation(uint64_t id, double score, double area,
                           bool is_crowd, bool ignore, bool lvis_mark)
            : id(id),
              score(score),
              area(area),
              is_crowd(is_crowd),
              ignore(ignore),
              lvis_mark(lvis_mark) {}

        uint64_t id;     // annotation id
        double score;    // confidence score
        double area;     // bounding box area
        bool is_crowd;   // crowd annotation
        bool ignore;     // ignore annotation
        bool lvis_mark;  // lvis mark
};

// Data for storing object match annotation results for each
// detection-ground truth pair with corresponding IoU
struct MatchedAnnotation {
        MatchedAnnotation(uint64_t dt_id, uint64_t gt_id, double iou)
            : dt_id(dt_id), gt_id(gt_id), iou(iou) {}

        uint64_t dt_id;
        uint64_t gt_id;
        double iou;
};

}  // namespace COCOeval
}  // namespace coco_eval
