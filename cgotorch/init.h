// C bindings for torch::nn::init.
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

#pragma once

#include "cgotorch/torchdef.h"

#ifdef __cplusplus
extern "C" {
#endif

const char* Torch_NN_Init_Zeros_(Tensor* tensor);
const char* Torch_NN_Init_Ones_(Tensor* tensor);
const char* Torch_NN_Init_Uniform_(Tensor* tensor, double low, double high);
const char* Torch_NN_Init_Normal_(Tensor* tensor, double mean, double std);
// const char* Torch_NN_Init_CalculateFanInAndFanOut(int64_t* fan_in, int64_t* fan_out, Tensor tensor);
// const char* Torch_NN_Init_KaimingUniform_(Tensor* tensor, double a, const char* fan_mod, const char* non_linearity);

#ifdef __cplusplus
}
#endif
