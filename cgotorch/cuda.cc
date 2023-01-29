// C bindings for torch::cuda.
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

#ifdef WITH_CUDA
#include "c10/cuda/CUDAStream.h"
#endif

#include "cgotorch/cuda.h"
#include "cgotorch/try_catch_return_error_string.hpp"

bool Torch_CUDA_IsAvailable() { return torch::cuda::is_available(); }

bool Torch_CUDA_IsCUDNNAvailable() { return torch::cuda::cudnn_is_available(); }

// const char* Torch_CUDA_GetCurrentCUDAStream(CUDAStream* stream, Device* device) {
//     return try_catch_return_error_string([&]() {
// #ifdef WITH_CUDA
//         *stream = static_cast<void *>(new at::cuda::CUDAStream(at::cuda::getCurrentCUDAStream((*device)->index())));
// #else
//         throw std::runtime_error("CUDA API needs -DWITH_CUDA when building libcgotorch");
// #endif
//     });
// }

// const char* Torch_CUDA_GetCUDAStreamFromPool(CUDAStream* stream, Device* device) {
//     return try_catch_return_error_string([&]() {
// #ifdef WITH_CUDA
//         *stream = static_cast<void*>(new at::cuda::CUDAStream(at::cuda::getStreamFromPool(false /**isHighPriority**/, (*device)->index())));
// #else
//         throw std::runtime_error("CUDA API needs -DWITH_CUDA when building libcgotorch");
// #endif
//     });
// }

// const char* Torch_CUDA_SetCurrentCUDAStream(CUDAStream stream) {
//     return try_catch_return_error_string([&]() {
// #ifdef WITH_CUDA
//         at::cuda::setCurrentCUDAStream(*static_cast<at::cuda::CUDAStream* >(stream));
// #else
//         throw std::runtime_error("CUDA API needs -DWITH_CUDA when building libcgotorch");
// #endif
//     });
// }

// const char* Torch_CUDA_Synchronize(CUDAStream stream) {
//     return try_catch_return_error_string([&]() {
// #ifdef WITH_CUDA
//         static_cast<at::cuda::CUDAStream* >(stream)->synchronize();
// #else
//         throw std::runtime_error("CUDA API needs -DWITH_CUDA when building libcgotorch");
// #endif
//     });
// }

// const char* Torch_CUDA_Query(CUDAStream stream, int8_t* result) {
//     return try_catch_return_error_string([&]() {
// #ifdef WITH_CUDA
//         *result = static_cast<at::cuda::CUDAStream* >(stream)->query() ? 1 : 0;
// #else
//         throw std::runtime_error("CUDA API needs -DWITH_CUDA when building libcgotorch");
// #endif
//     });
// }
