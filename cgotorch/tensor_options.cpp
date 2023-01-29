// C bindings for at::TensorOptions.
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

#include "cgotorch/tensor_options.h"
#include "cgotorch/try_catch_return_error_string.hpp"

const char* Torch_TensorOptions(TensorOptions* options) {
    return try_catch_return_error_string([&]() {
        *options = new at::TensorOptions();
    });
}

void Torch_TensorOptions_Free(TensorOptions options) { delete options; }

const char* Torch_TensorOptions_Dtype(TensorOptions* output, TensorOptions options, int8_t dtype) {
    return try_catch_return_error_string([&]() {
        *output = new at::TensorOptions(options->dtype(static_cast<at::ScalarType>(dtype)));
    });
}

const char* Torch_TensorOptions_Layout(TensorOptions* output, TensorOptions options, int8_t layout) {
    return try_catch_return_error_string([&]() {
        *output = new at::TensorOptions(options->layout(static_cast<c10::Layout>(layout)));
    });
}

const char* Torch_TensorOptions_MemoryFormat(TensorOptions* output, TensorOptions options, int8_t memory_format) {
    return try_catch_return_error_string([&]() {
        *output = new at::TensorOptions(options->memory_format(static_cast<c10::MemoryFormat>(memory_format)));
    });
}

const char* Torch_TensorOptions_Device(TensorOptions* output, TensorOptions options, Device device) {
    return try_catch_return_error_string([&]() {
        *output = new at::TensorOptions(options->device(*device));
    });
}

const char* Torch_TensorOptions_RequiresGrad(TensorOptions* output, TensorOptions options, bool requires_grad) {
    return try_catch_return_error_string([&]() {
        *output = new at::TensorOptions(options->requires_grad(requires_grad));
    });
}

const char* Torch_TensorOptions_PinnedMemory(TensorOptions* output, TensorOptions options, bool pinned_memory) {
    return try_catch_return_error_string([&]() {
        *output = new at::TensorOptions(options->pinned_memory(pinned_memory));
    });
}
