#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <exception>

#ifdef __cplusplus

namespace vision {
namespace ops {

at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold);

} // namespace ops
} // namespace vision

#endif
