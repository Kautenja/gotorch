// C bindings for torch::optim.
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

#include "cgotorch/optim.h"

#include <vector>

// Optimizer SGD(double learning_rate, double momentum, double dampening,
//               double weight_decay, int64_t nesterov) {
//   return new torch::optim::SGD(std::vector<torch::optim::OptimizerParamGroup>{},
//                                torch::optim::SGDOptions(learning_rate)
//                                    .momentum(momentum)
//                                    .dampening(dampening)
//                                    .weight_decay(weight_decay)
//                                    .nesterov(nesterov));
// }

// Optimizer Adam(double learning_rate, double beta1, double beta2,
//                double weight_decay) {
//   auto options = torch::optim::AdamOptions(learning_rate)
//                      .betas(std::make_tuple(beta1, beta2))
//                      .weight_decay(weight_decay);
//   return new torch::optim::Adam(std::vector<torch::Tensor>(), options);
// }

// void Optimizer_ZeroGrad(Optimizer opt) { opt->zero_grad(); }

// void Optimizer_Step(Optimizer opt) { opt->step(); }

// void Optimizer_AddParameters(Optimizer opt, Tensor* tensors, int64_t length) {
//   std::vector<torch::Tensor> params;
//   while (params.size() < length) params.push_back(**tensors++);
//   opt->add_param_group({params});
// }

// void Optimizer_SetLR(Optimizer opt, double learning_rate) {
//   if (dynamic_cast<torch::optim::SGD*>(opt)) {
//     for (auto& pg : opt->param_groups()) {
//       static_cast<torch::optim::SGDOptions*>(&pg.options())->lr(learning_rate);
//     }
//   } else if (dynamic_cast<torch::optim::Adam*>(opt)) {
//     for (auto& pg : opt->param_groups()) {
//       static_cast<torch::optim::AdamOptions*>(&pg.options())->lr(learning_rate);
//     }
//   }
// }

// void Optimizer_Close(Optimizer opt) { delete opt; }
