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

#include <torch/script.h>
#include <string>
#include "cgotorch/jit.h"
#include "cgotorch/try_catch_return_error_string.hpp"

const char* Torch_Jit_Load(JitModule* module, const char* path, Device device) {
    return try_catch_return_error_string([&]() {
        *module = new torch::jit::Module(torch::jit::load(std::string(path), *device));
    });
}

void Torch_Jit_Module_Free(JitModule module) { delete module; }

const char* Torch_Jit_Module_Save(const char* path, JitModule module) {
    return try_catch_return_error_string([&]() { module->save(path); });
}

const char* Torch_Jit_Module_String(JitModule module) {
    std::string str = module->dump_to_str(false, false, false);
    char* output = reinterpret_cast<char*>(malloc(str.size() + 1));
    snprintf(output, str.size() + 1, "%s", str.c_str());
    return output;
}

const char* Torch_Jit_Module_IsTraining(bool* output, JitModule module) {
    return try_catch_return_error_string([&]() { *output = module->is_training(); });
}

const char* Torch_Jit_Module_IsOptimized(bool* output, JitModule module) {
    return try_catch_return_error_string([&]() { *output = module->is_optimized(); });
}

const char* Torch_Jit_Module_SetOptimized(JitModule module, bool mode) {
    return try_catch_return_error_string([&]() { module->set_optimized(mode); });
}

const char* Torch_Jit_Module_Train(JitModule module, bool mode) {
    return try_catch_return_error_string([&]() { module->train(mode); });
}

const char* Torch_Jit_Module_Eval(JitModule module) {
    return try_catch_return_error_string([&]() { module->eval(); });
}

const char* Torch_Jit_Module_CastTo(JitModule module, int8_t dtype) {
    return try_catch_return_error_string([&]() { module->to(static_cast<at::ScalarType>(dtype)); });
}

const char* Torch_Jit_Module_CopyTo(JitModule module, Device device) {
    return try_catch_return_error_string([&]() { module->to(*device); });
}

// TODO: Torch_Jit_Module_Copy
// TODO: Torch_Jit_Module_DeepCopy
// TODO: Torch_Jit_Module_Clone

const char* Torch_Jit_Module_Forward(
    IValue* output,
    JitModule module,
    IValue* inputs,
    int64_t num_inputs
) {
    return try_catch_return_error_string([&]() {
        std::vector<torch::jit::IValue> ivalues;
        for (int i = 0; i < num_inputs; ++i) ivalues.push_back(*inputs[i]);
        *output = new torch::IValue(module->forward(ivalues));
        // *output = new torch::IValue(module->forward(
        //     std::vector<torch::jit::IValue>(inputs, inputs + num_inputs)
        // ));
    });
}
