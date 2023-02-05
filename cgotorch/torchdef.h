// C bindings of libtorch structures.
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

#include <stdbool.h>  // maps "_Bool" to "bool" and defines "false" & "true"
#include <stdint.h>   // Defines the standard width integer types (int32_t, ...)

#ifdef __cplusplus
#include <torch/torch.h>
#include <vector>
extern "C" {
typedef at::Tensor* Tensor;
typedef c10::TensorOptions* TensorOptions;
typedef torch::optim::Optimizer* Optimizer;
typedef torch::data::datasets::MNIST* MNIST;
typedef torch::Device* Device;
typedef std::vector<char>* ByteBuffer;
typedef torch::jit::Module* JitModule;
typedef torch::IValue* IValue;
#else
typedef void* Tensor;
typedef void* TensorOptions;
typedef void* Optimizer;
typedef void* MNIST;
typedef void* Device;
typedef void* ByteBuffer;
typedef void* JitModule;
typedef void* IValue;
#endif

typedef void* CUDAStream;

/// @brief Return the libtorch major version.
/// @returns the major version of libtorch (following semantic versioning.)
int64_t TorchMajorVersion();

/// @brief Return the libtorch minor version.
/// @returns the minor version of libtorch (following semantic versioning.)
int64_t TorchMinorVersion();

/// @brief Return the libtorch patch version.
/// @returns the patch version of libtorch (following semantic versioning.)
int64_t TorchPatchVersion();

/// @brief Set the random number generator seed.
/// @param seed The new seed to set the RNG to.
void ManualSeed(int64_t seed);

/// @brief Set the global gradient generation state.
/// @param value True to enable gradient generation globally, false to disable it.
void SetGradEnabled(bool value);

/// @brief Return the global gradient generation state.
/// @param value True if gradient generation is enabled, false otherwise.
bool IsGradEnabled();

/// @brief Set the number of threads used for intraop parallelism on CPU.
/// @param num_threads The number of threads to use.
void SetNumThreads(int32_t num_threads);

#ifdef __cplusplus
}
#endif
