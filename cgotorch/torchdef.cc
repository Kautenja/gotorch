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

#include "cgotorch/torchdef.h"

int64_t TorchMajorVersion() {
    return TORCH_VERSION_MAJOR;
}

int64_t TorchMinorVersion() {
    return TORCH_VERSION_MINOR;
}

int64_t TorchPatchVersion() {
    return TORCH_VERSION_PATCH;
}

void ManualSeed(int64_t seed) {
    torch::manual_seed(seed);
}

void SetGradEnabled(bool value) {
    at::GradMode::set_enabled(value);
}

bool IsGradEnabled() {
    return at::GradMode::is_enabled();
}

void SetNumThreads(int32_t num_threads) {
    torch::set_num_threads(num_threads);
}
