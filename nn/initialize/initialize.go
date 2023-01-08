// Go bindings for torch::nn::init.
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

package nn_initialize

// #cgo CFLAGS: -I ${SRCDIR}/../..
// #cgo LDFLAGS: -L ${SRCDIR}/../../build -Wl,-rpath ${SRCDIR}/../../build -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/../../build/libtorch/lib -Wl,-rpath ${SRCDIR}/../../build/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include <stdio.h>
// #include <stdlib.h>
// #include "cgotorch/cgotorch.h"
import "C"
import (
    // "unsafe"
    // "github.com/Kautenja/gotorch"
    // internal "github.com/Kautenja/gotorch/internal"
)

// func Zeros(a *torch.Tensor) {
//     if a == nil || a.T == nil {
//         panic("input tensor is nil")
//     }
//     internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Init_Zeros_((*C.Tensor)(a.T))))
// }

// func Ones(a *torch.Tensor) {
//     if a == nil || a.T == nil {
//         panic("input tensor is nil")
//     }
//     internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Init_Ones_((*C.Tensor)(a.T))))
// }

// func Uniform(a *torch.Tensor, low, high float64) {
//     if a == nil || a.T == nil {
//         panic("input tensor is nil")
//     }
//     internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Init_Uniform_(
//         (*C.Tensor)(a.T),
//         C.double(low),
//         C.double(high),
//     )))
// }

// func Normal(a *torch.Tensor, mean, std float64) {
//     if a == nil || a.T == nil {
//         panic("input tensor is nil")
//     }
//     internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Init_Normal_(
//         (*C.Tensor)(a.T),
//         C.double(mean),
//         C.double(std),
//     )))
// }

// // torch.nn.init._calculate_fan_in_and_fan_out
// func CalculateFanInAndFanOut(input torch.Tensor) (int64, int64) {
//     var fanIn, fanOut int64
//     internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Init_CalculateFanInAndFanOut(
//         (*C.int64_t)(unsafe.Pointer(&fanIn)),
//         (*C.int64_t)(unsafe.Pointer(&fanOut)),
//         C.Tensor(*input.T),
//     )))
//     return fanIn, fanOut
// }

// func KaimingUniform(input *torch.Tensor, a float64, fanMode string,
//     nonLinearity string) {
//     if input == nil || input.T == nil {
//         panic("Normal: input tensor is nil")
//     }
//     internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Init_KaimingUniform_(
//         (*C.Tensor)(input.T),
//         C.double(a),
//         C.CString(fanMode),
//         C.CString(nonLinearity),
//     )))
// }
