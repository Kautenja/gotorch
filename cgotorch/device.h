// C bindings for torch::Device.
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

/// @brief Create a new device with the given descriptive name.
/// @param device A pointer to the device to create.
/// @param device_name The descriptive name of the device.
const char* Torch_Device(Device *device, const char* device_name);

/// @brief Free the heap memory used to hold the given Device.
/// @param device The Device to free from the heap.
void Torch_Device_Free(Device device);

/// @brief Return true if the given string refers to a valid compute device.
/// @param device_name The name of the device to check
/// @returns True if the `device_name` is a valid torch::device
bool Torch_IsDevice(const char* device_name);

#ifdef __cplusplus
}
#endif
