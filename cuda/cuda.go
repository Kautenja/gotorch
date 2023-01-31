// Go bindings for torch::cuda.
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

package torch_cuda

// #cgo CPPFLAGS: -I/usr/local/include -I/usr/local/include/cgotorch
// #cgo LDFLAGS: -L/usr/local/lib -lc10 -ltorch_cpu -ltorch -lcgotorch
// #include <stdio.h>
// #include <stdlib.h>
// #include "cgotorch/cgotorch.h"
import "C"
import (
    // "unsafe"
    // torch "github.com/Kautenja/gotorch"
    // internal "github.com/Kautenja/gotorch/internal"
)

// Return true if CUDA is available
func IsAvailable() bool {
    return bool(C.Torch_CUDA_IsAvailable())
}

// Return true if cuDNN is available
func IsCUDNNAvailable() bool {
    return bool(C.Torch_CUDA_IsCUDNNAvailable())
}

// // CUDAStream struct wrapped Nvidia CUDA Stream
// type CUDAStream struct {
//     P C.CUDAStream
// }

// // Query returns true if all tasks completed on this CUDA stream
// func (s CUDAStream) Query() bool {
//     var b int8
//     internal.PanicOnCException(unsafe.Pointer(C.Torch_CUDA_Query(s.P, (*C.int8_t)(&b))))
//     return b != 0
// }

// // Synchronize wait until all tasks completed on this CUDA stream
// func (s CUDAStream) Synchronize() {
//     internal.PanicOnCException(unsafe.Pointer(C.Torch_CUDA_Synchronize(s.P)))
// }

// // GetCurrentCUDAStream returns the current stream on device
// func GetCurrentCUDAStream(device torch.Device) CUDAStream {
//     var stream C.CUDAStream
//     internal.PanicOnCException(unsafe.Pointer(C.Torch_CUDA_GetCurrentCUDAStream(&stream, (*C.Device)(&device.T))))
//     return CUDAStream{stream}
// }

// // SetCurrentCUDAStream set stream as the current CUDA stream
// func SetCurrentCUDAStream(stream CUDAStream) {
//     internal.PanicOnCException(unsafe.Pointer(C.Torch_CUDA_SetCurrentCUDAStream(stream.P)))
// }

// // NewCUDAStream returns a new CUDA stream from the pool
// func NewCUDAStream(device torch.Device) CUDAStream {
//     var stream C.CUDAStream
//     internal.PanicOnCException(unsafe.Pointer(C.Torch_CUDA_GetCUDAStreamFromPool(&stream, (*C.Device)(&device.T))))
//     return CUDAStream{stream}
// }
