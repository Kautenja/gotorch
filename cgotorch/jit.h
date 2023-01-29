// C bindings for torch::jit.
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

// TODO: const char* Torch_Jit_OptimizeForInference

/// @brief Load a module trace from the file-system.
/// @param output A pointer to a pointer to initialize with the module.
/// @param path A path to a traced module on the file-system.
/// @param device The device to load the module onto.
/// @returns A pointer to a string error message (nullptr on success.)
const char* Torch_Jit_Load(JitModule* output, const char* path, Device device);

/// @brief Free the heap memory used to hold the given JitModule.
/// @param module The JitModule to free from the heap.
void Torch_Jit_Module_Free(JitModule module);

/// @brief Save a module trace to the file-system.
/// @param path A path on the file-system to save the module to.
/// @param module The module to save.
/// @returns A pointer to a string error message (nullptr on success.)
const char* Torch_Jit_Module_Save(const char* path, JitModule module);

/// @brief Convert the module to a string representation.
/// @param a The module to convert into a string
/// @returns A character pointer to a string representation.
/// @details
/// Ownership of the output character buffer is transferred to the caller. The
/// memory should be released using `std::free` when done. **DO NOT** use the
/// `delete` operator to free the memory, it is allocated in C using `malloc`.
const char* Torch_Jit_Module_String(JitModule module);

/// @brief Check if a module has training features enabled.
/// @param output true if training features are enabled, false otherwise.
/// @param module The module to check the training setting of.
/// @returns A pointer to a string error message (nullptr on success.)
const char* Torch_Jit_Module_IsTraining(bool* output, JitModule module);

/// @brief Check if a module has JIT optimization features enabled.
/// @param output true if JIT optimization is enabled, false otherwise.
/// @param module The module to check the optimization setting of.
/// @returns A pointer to a string error message (nullptr on success.)
const char* Torch_Jit_Module_IsOptimized(bool* output, JitModule module);

/// @brief Enable/disable JIT optimization features for the module.
/// @param module The module to set the optimization setting of.
/// @param mode True to enable JIT optimizations, false to disable them.
/// @returns A pointer to a string error message (nullptr on success.)
const char* Torch_Jit_Module_SetOptimized(JitModule module, bool mode);

/// @brief Enable/disable training features for the module.
/// @param module The module to set the training setting of.
/// @param mode True to use training features, false to disable them.
/// @returns A pointer to a string error message (nullptr on success.)
const char* Torch_Jit_Module_Train(JitModule module, bool mode);

/// @brief Set the module to evaluation (e.g., inference) mode.
/// @param module The module to set to eval mode.
/// @returns A pointer to a string error message (nullptr on success.)
/// @details
/// This method is the same as calling `Torch_Jit_Module_Train(module, false);`
const char* Torch_Jit_Module_Eval(JitModule module);

/// @brief Cast the tensor to the given data-type in-place.
/// @param module The module to cast to a different data-type.
/// @param dtype The data-type to cast the module's parameters to.
/// @returns A pointer to a string error message (nullptr on success.)
const char* Torch_Jit_Module_CastTo(JitModule module, int8_t dtype);

/// @brief Copy the tensor to the given device in-place.
/// @param module The module to copy to a different compute accelerator.
/// @param device The accelerator to copy the parameter data onto.
/// @returns A pointer to a string error message (nullptr on success.)
const char* Torch_Jit_Module_CopyTo(JitModule module, Device device);

// TODO: const char* Torch_Jit_Module_Copy(JitModule* output, JitModule module)
// TODO: const char* Torch_Jit_Module_DeepCopy(JitModule* output, JitModule module)
// TODO: const char* Torch_Jit_Module_Clone(JitModule* output, JitModule module)

/// @brief Forward pass data through a JIT module.s
/// @param module The module to forward pass through.
/// @param outputs A pointer to a buffer to populate with module outputs.
/// @param inputs The inputs to the module.
/// @param num_inputs The number of tensor inputs.
/// @returns A pointer to a string error message (nullptr on success.)
const char* Torch_Jit_Module_Forward(
    IValue* output,
    JitModule module,
    IValue* inputs,
    int64_t num_inputs
);

#ifdef __cplusplus
}
#endif
