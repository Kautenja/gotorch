// C bindings for torch::nn::init.
//
// Copyright (c) 2022 Christian Kauten
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

#include "cgotorch/init.h"
#include <string>
#include <unordered_map>
#include "cgotorch/try_catch_return_error_string.hpp"

const char* Torch_NN_Init_Zeros_(Tensor* tensor) {
    return try_catch_return_error_string([&] () {
        torch::nn::init::zeros_(**tensor);
    });
}

const char* Torch_NN_Init_Ones_(Tensor* tensor) {
    return try_catch_return_error_string([&] () {
        torch::nn::init::ones_(**tensor);
    });
}

const char* Torch_NN_Init_Uniform_(Tensor* tensor, double low, double high) {
    return try_catch_return_error_string([&] () {
        torch::nn::init::uniform_(**tensor, low, high);
    });
}

const char* Torch_NN_Init_Normal_(Tensor* tensor, double mean, double std) {
    return try_catch_return_error_string([&] () {
        torch::nn::init::normal_(**tensor, mean, std);
    });
}

// const char* Torch_NN_Init_CalculateFanInAndFanOut(int64_t* fan_in, int64_t* fan_out, Tensor tensor) {
//     return try_catch_return_error_string([&] () {
//         const auto &res = torch::nn::init::_calculate_fan_in_and_fan_out(*tensor);
//         *fan_in = std::get<0>(res);
//         *fan_out = std::get<1>(res);
//     });
// }

// std::unordered_map<std::string, torch::nn::init::FanModeType> fan_mode_map = {
//     {"fan_in", torch::kFanIn},
//     {"fan_out", torch::kFanOut},
// };

// std::unordered_map<std::string, torch::nn::init::NonlinearityType>
//     non_linearity_map = {
//         {"relu", torch::kReLU},
//         {"leaky_relu", torch::kLeakyReLU},
//         {"tanh", torch::kTanh},
//         {"sigmoid", torch::kSigmoid},
//         {"linear", torch::kLinear},
//         {"conv1d", torch::kConv1D},
//         {"conv2d", torch::kConv2D},
//         {"conv3d", torch::kConv3D},
//         {"conv_transpose1d", torch::kConvTranspose1D},
//         {"conv_transpose2d", torch::kConvTranspose2D},
//         {"conv_transpose3d", torch::kConvTranspose3D},
// };

// const char* Torch_NN_Init_KaimingUniform_(
//     Tensor* tensor, double a, const char* fan_mod, const char* non_linearity
// ) {
//     return try_catch_return_error_string([&] () {
//         torch::nn::init::kaiming_uniform_(**tensor, a,
//             fan_mode_map[std::string(fan_mod)],
//             non_linearity_map[std::string(non_linearity)]
//         );
//     });
// }
