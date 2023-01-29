// C bindings for torch::nn::functional.
//
// Copyright (c) 2023 Christian Kauten
// Copyright (c) 2022 Sensory, Inc.
// Copyright (c) 2020 GoTorch Authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXTERNRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include <string>
#include <unordered_map>
#include <vector>
#include "cgotorch/functional.h"
#include "cgotorch/try_catch_return_error_string.hpp"

// torch::nn::functional::adaptive_avg_pool1d

// torch::nn::functional::adaptive_avg_pool2d
const char *Torch_NN_Functional_AdaptiveAvgPool2d(
  Tensor *result,
  Tensor input,
  int64_t *output_size_data,
  int64_t output_size_len
) {
  return try_catch_return_error_string([&]() {
    *result = new at::Tensor(torch::nn::functional::adaptive_avg_pool2d(*input,
      torch::nn::functional::AdaptiveAvgPool2dFuncOptions(torch::IntArrayRef(output_size_data, output_size_len))));
  });
}

// torch::nn::functional::adaptive_avg_pool3d
// torch::nn::functional::adaptive_max_pool1d
// torch::nn::functional::adaptive_max_pool2d
// torch::nn::functional::adaptive_max_pool2d_with_indices
// torch::nn::functional::adaptive_max_pool3d
// torch::nn::functional::adaptive_max_pool3d_with_indices
// torch::nn::functional::affine_grid
// torch::nn::functional::alpha_dropout
// torch::nn::functional::avg_pool1d
// torch::nn::functional::avg_pool2d
// torch::nn::functional::avg_pool3d

// torch::nn::functional::batch_norm
const char *Torch_NN_Functional_BatchNorm(
  Tensor *result,
  Tensor input,
  Tensor weight,
  Tensor bias,
  Tensor running_mean,
  Tensor running_var,
  int8_t training,
  double momentum,
  double eps
) {
  return try_catch_return_error_string([&](){
    *result = new at::Tensor(torch::nn::functional::batch_norm(*input,
      (running_mean ? *running_mean : at::Tensor()),
      (running_var ? *running_var : at::Tensor()),
      torch::nn::functional::BatchNormFuncOptions()
        .weight(weight ? *weight : at::Tensor())
        .bias(bias ? *bias : at::Tensor())
        .training(training)
        .momentum(momentum)
        .eps(eps))
    );
  });
}

// torch::nn::functional::bilinear

// torch::nn::functional::binary_cross_entropy
const char *Torch_NN_Functional_BinaryCrossEntropy(
  Tensor *result,
  Tensor input,
  Tensor target,
  Tensor weight,
  const char *reduction
) {
  static std::unordered_map<
    std::string, torch::nn::BCELossOptions::reduction_t
  > reduce_map = {
    {"none", torch::kNone},
    {"mean", torch::kMean},
    {"sum",  torch::kSum },
  };
  return try_catch_return_error_string([&](){
    *result = new at::Tensor(torch::nn::functional::binary_cross_entropy(*input, *target,
      torch::nn::functional::BinaryCrossEntropyFuncOptions()
        .weight((weight ? *weight : torch::Tensor()))
        .reduction(reduce_map[std::string(reduction)]))
    );
  });
}

// torch::nn::functional::binary_cross_entropy_with_logits
// torch::nn::functional::celu
// torch::nn::functional::conv1d

// torch::nn::functional::conv2d
const char *Torch_NN_Functional_Conv2d(
  Tensor *result,
  Tensor input,
  Tensor weight,
  Tensor bias,
  int64_t *stride_data,
  int64_t stride_len,
  int64_t *padding_data,
  int64_t padding_len,
  int64_t *dilation_data,
  int64_t dilation_len,
  int64_t groups
) {
  return try_catch_return_error_string([&](){
    *result = new at::Tensor(torch::nn::functional::conv2d(*input, *weight,
      torch::nn::functional::Conv2dFuncOptions()
        .bias(bias ? *bias : at::Tensor())
        .stride(torch::IntArrayRef(stride_data, stride_len))
        .padding(torch::IntArrayRef(padding_data, padding_len))
        .dilation(torch::IntArrayRef(dilation_data, dilation_len))
        .groups(groups)
    ));
  });
}

// torch::nn::functional::conv3d
// torch::nn::functional::conv_transpose1d

// torch::nn::functional::conv_transpose2d
const char *Torch_NN_Functional_ConvTranspose2d(
  Tensor *result,
  Tensor input,
  Tensor weight,
  Tensor bias,
  int64_t *stride_data,
  int64_t stride_len,
  int64_t *padding_data,
  int64_t padding_len,
  int64_t *output_padding_data,
  int64_t output_padding_len,
  int64_t groups,
  int64_t *dilation_data,
  int64_t dilation_len
) {
  return try_catch_return_error_string([&](){
    *result = new at::Tensor(torch::nn::functional::conv_transpose2d(*input, *weight,
      torch::nn::functional::ConvTranspose2dFuncOptions()
        .bias(bias ? *bias : at::Tensor())
        .stride(torch::IntArrayRef(stride_data, stride_len))
        .padding(torch::IntArrayRef(padding_data, padding_len))
        .output_padding(torch::IntArrayRef(output_padding_data, output_padding_len))
        .groups(groups)
        .dilation(torch::IntArrayRef(dilation_data, dilation_len))
    ));
  });
}

// torch::nn::functional::conv_transpose3d
// torch::nn::functional::cosine_embedding_loss
// torch::nn::functional::cosine_similarity

// torch::nn::functional::cross_entropy
const char *Torch_NN_Functional_CrossEntropy(
  Tensor *result,
  Tensor input,
  Tensor target,
  Tensor weight,
  int64_t ignore_index,
  const char *reduction
) {
  static std::unordered_map<
    std::string, torch::nn::CrossEntropyLossOptions::reduction_t
  > reduce_map = {
    {"none", torch::kNone},
    {"mean", torch::kMean},
    {"sum",  torch::kSum},
  };
  return try_catch_return_error_string([&](){
    *result = new at::Tensor(torch::nn::functional::cross_entropy(*input, *target,
      torch::nn::functional::CrossEntropyFuncOptions()
        .weight((weight ? *weight : torch::Tensor()))
        .ignore_index(ignore_index)
        .reduction(reduce_map[std::string(reduction)])
    ));
  });
}

// torch::nn::functional::ctc_loss
// torch::nn::functional::dropout
// torch::nn::functional::dropout2d
// torch::nn::functional::dropout3d
// torch::nn::functional::elu
// torch::nn::functional::embedding
// torch::nn::functional::embedding_bag
// torch::nn::functional::feature_alpha_dropout
// torch::nn::functional::fold
// torch::nn::functional::fractional_max_pool2d
// torch::nn::functional::fractional_max_pool2d_with_indices
// torch::nn::functional::fractional_max_pool3d
// torch::nn::functional::fractional_max_pool3d_with_indices
// torch::nn::functional::gelu
// torch::nn::functional::glu
// torch::nn::functional::grid_sample
// torch::nn::functional::group_norm
// torch::nn::functional::gumbel_softmax
// torch::nn::functional::hardshrink
// torch::nn::functional::hardtanh
// torch::nn::functional::hinge_embedding_loss
// torch::nn::functional::huber_loss
// torch::nn::functional::instance_norm

/// @brief Convert an interpolate mode to the associated torch structure.
/// @param mode The interpolation mode to map to a `mode_t`.
/// @returns A `mode_t` that can be passed to `InterpolateFuncOptions`.
inline torch::nn::functional::InterpolateFuncOptions::mode_t InterpolateModeToModeT(const InterpolateMode& mode) {
  switch (mode) {
    case InterpolateNearest:      return torch::kNearest;
    case InterpolateLinear:       return torch::kLinear;
    case InterpolateBilinear:     return torch::kBilinear;
    case InterpolateBicubic:      return torch::kBicubic;
    case InterpolateTrilinear:    return torch::kTrilinear;
    case InterpolateArea:         return torch::kArea;
    case InterpolateNearestExact: return torch::kNearestExact;
  }
  throw std::runtime_error("interpolation mode " + std::to_string(mode) + " is not supported");
}

const char* Torch_NN_Functional_InterpolateSize(
  Tensor* result,
  Tensor input,
  int64_t* size,
  int64_t dims,
  int64_t mode,
  bool align_corners,
  bool antialias
) {
  return try_catch_return_error_string([&]() {
    torch::nn::functional::InterpolateFuncOptions options;
    options = options.size(std::vector<int64_t>(size, size + dims));
    options = options.mode(InterpolateModeToModeT(static_cast<InterpolateMode>(mode)));
    if (align_corners) options = options.align_corners(align_corners);
    options = options.antialias(antialias);
    *result = new at::Tensor(torch::nn::functional::interpolate(*input, options));
  });
}

const char* Torch_NN_Functional_InterpolateScale(
  Tensor* result,
  Tensor input,
  double* scale,
  int64_t dims,
  int64_t mode,
  bool align_corners,
  bool antialias
) {
  return try_catch_return_error_string([&](){
    torch::nn::functional::InterpolateFuncOptions options;
    options = options.scale_factor(std::vector<double>(scale, scale + dims));
    options = options.mode(InterpolateModeToModeT(static_cast<InterpolateMode>(mode)));
    if (align_corners) options = options.align_corners(align_corners);
    options = options.antialias(antialias);
    *result = new at::Tensor(torch::nn::functional::interpolate(*input, options));
  });
}

// torch::nn::functional::kl_div
// torch::nn::functional::l1_loss
// torch::nn::functional::layer_norm

// torch::nn::functional::leaky_relu
const char *Torch_NN_Functional_LeakyRelu(
  Tensor *result,
  Tensor input,
  double negative_slope,
  bool inplace
) {
  return try_catch_return_error_string([&](){
    *result = new at::Tensor(torch::nn::functional::leaky_relu(*input,
      torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope).inplace(inplace)
    ));
  });
}

// torch::nn::functional::linear
const char *Torch_NN_Functional_Linear(
  Tensor *result,
  Tensor input,
  Tensor weight,
  Tensor bias
) {
  return try_catch_return_error_string([&](){
    *result = new at::Tensor(torch::nn::functional::linear(*input, *weight,
      (bias ? *bias : torch::Tensor())
    ));
  });
}

// torch::nn::functional::local_response_norm

// torch::nn::functional::log_softmax
const char* Torch_NN_Functional_LogSoftmax(
  Tensor* result,
  Tensor input,
  int64_t dim
) {
  return try_catch_return_error_string([&] () {
    *result = new at::Tensor(input->log_softmax(dim));
  });
}

// torch::nn::functional::logsigmoid
// torch::nn::functional::lp_pool1d
// torch::nn::functional::lp_pool2d
// torch::nn::functional::margin_ranking_loss
// torch::nn::functional::max_pool1d
// torch::nn::functional::max_pool1d_with_indices

// torch::nn::functional::max_pool2d
const char *Torch_NN_Functional_MaxPool2d(
  Tensor *result,
  Tensor input,
  int64_t *kernel_data,
  int64_t kernel_len,
  int64_t *stride_data,
  int64_t stride_len,
  int64_t *padding_data,
  int64_t padding_len,
  int64_t *dilation_data,
  int64_t dilation_len,
  int8_t ceil_mode
) {
  return try_catch_return_error_string([&](){
    auto out = torch::nn::functional::max_pool2d(*input,
      torch::nn::functional::MaxPool2dFuncOptions(torch::IntArrayRef(kernel_data, kernel_len))
        .stride(torch::IntArrayRef(stride_data, stride_len))
        .padding(torch::IntArrayRef(padding_data, padding_len))
        .dilation(torch::IntArrayRef(dilation_data, dilation_len))
        .ceil_mode(ceil_mode)
    );
    *result = new at::Tensor(out);
  });
}

// torch::nn::functional::max_pool2d_with_indices
// torch::nn::functional::max_pool3d
// torch::nn::functional::max_pool3d_with_indices
// torch::nn::functional::max_unpool1d
// torch::nn::functional::max_unpool2d
// torch::nn::functional::max_unpool3d
// torch::nn::functional::mish
// torch::nn::functional::mse_loss
// torch::nn::functional::multi_head_attention_forward
// torch::nn::functional::multi_margin_loss
// torch::nn::functional::multilabel_margin_loss
// torch::nn::functional::multilabel_soft_margin_loss

// torch::nn::functional::nll_loss
const char *Torch_NN_Functional_NllLoss(
  Tensor *result,
  Tensor input,
  Tensor target,
  Tensor weight,
  int64_t ignore_index,
  const char *reduction
) {
  static std::unordered_map<std::string, torch::nn::NLLLossOptions::reduction_t>
      reduce_map = {
          {"none", torch::kNone},
          {"mean", torch::kMean},
          {"sum", torch::kSum},
      };
  return try_catch_return_error_string([&](){
    *result = new at::Tensor(torch::nn::functional::nll_loss(*input, *target,
      torch::nn::functional::NLLLossFuncOptions()
        .weight((weight ? *weight : torch::Tensor()))
        .ignore_index(ignore_index)
        .reduction(reduce_map[std::string(reduction)])
    ));
  });
}

// torch::nn::functional::normalize
const char* Torch_NN_Functional_Normalize(
  Tensor* result,
  Tensor input,
  double p,
  int64_t dim,
  double eps
) {
  return try_catch_return_error_string([&](){
    *result = new at::Tensor(torch::nn::functional::normalize(*input,
      torch::nn::functional::NormalizeFuncOptions().p(p).dim(dim).eps(eps)
    ));
  });
}

// torch::nn::functional::one_hot

/// @brief Convert a padding mode to the associated torch structure.
/// @param mode The padding mode to map to a `mode_t`.
/// @returns A `mode_t` that can be passed to `PadFuncOptions`.
inline torch::nn::functional::PadFuncOptions::mode_t PadModeToModeT(const PadMode& mode) {
  switch (mode) {
    case PadConstant:   return torch::kConstant;
    case PadReflect:    return torch::kReflect;
    case PadReplicate:  return torch::kReplicate;
    case PadCircular:   return torch::kCircular;
  }
  throw std::runtime_error("padding mode " + std::to_string(mode) + " is not supported");
}

const char* Torch_NN_Functional_Pad(
    Tensor* result,
    Tensor input,
    int64_t* padding,
    int64_t padding_length,
    int64_t mode,
    double value = 0
) {
  return try_catch_return_error_string([&](){
    *result = new at::Tensor(torch::nn::functional::pad(*input,
      torch::nn::functional::PadFuncOptions(std::vector<int64_t>(padding, padding + padding_length))
        .mode(PadModeToModeT(static_cast<PadMode>(mode)))
        .value(value)
    ));
  });
}

// torch::nn::functional::pairwise_distance
// torch::nn::functional::pdist
// torch::nn::functional::pixel_shuffle
// torch::nn::functional::pixel_unshuffle
// torch::nn::functional::poisson_nll_loss
// torch::nn::functional::prelu

// torch::nn::functional::relu
const char *Torch_NN_Functional_Relu(Tensor *result, Tensor input, bool inplace) {
  return try_catch_return_error_string([&](){
    *result = new at::Tensor(torch::nn::functional::relu(*input,
      torch::nn::functional::ReLUFuncOptions(inplace)
    ));
  });
}

// torch::nn::functional::relu6
// torch::nn::functional::rrelu
// torch::nn::functional::selu
// torch::nn::functional::silu
// torch::nn::functional::smooth_l1_loss
// torch::nn::functional::soft_margin_loss

// torch::nn::functional::softmax
const char* Torch_NN_Functional_Softmax(
  Tensor* result,
  Tensor input,
  int64_t dim
) {
  return try_catch_return_error_string([&] () {
    *result = new at::Tensor(input->softmax(dim));
  });
}

// torch::nn::functional::softmin
// torch::nn::functional::softplus
// torch::nn::functional::softshrink
// torch::nn::functional::softsign
// torch::nn::functional::tanhshrink
// torch::nn::functional::threshold
// torch::nn::functional::triplet_margin_loss
// torch::nn::functional::triplet_margin_with_distance_loss
// torch::nn::functional::unfold
