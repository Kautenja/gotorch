// A garbage collection module for GoTorch.
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

// GoTorch should allow goroutines other than the main goroutine to create
// tensors whose lifecycle last for more than one iteration, for example, we
// have to cache tensors in data loading goroutines because they run faster.
//
// To meet this goal, we have to make sure a `torch.GC()` function call in the
// main goroutine know which tensors are created by the main goroutine, and only
// wait for tensors created in the main goroutine to be freed.
//
// Actually, we need a "goroutine local storage" mechanism to distinguish the
// main goroutine and the data loading goroutine. However, Go doesn't provide
// an official "goroutine local storage", and the official `context` package
// will impose additional parameters to user API, thus make the API harder to
// use.
//
// Recall that we've already locked the main goroutine to a fixed OS thread in
// the `init` function in device.go, we can use a C++ `thread_local` to solve
// the problem.
//
// The three functions `C.GCPrepared()`, `C.PrepareGC()`, and `C.FinishGC()` are
// defined in cgotorch/memory.cc, they use the C++ variable
// `thread_local bool gcPrepared`
// to help control the behavior of the function `torch.GC()` as described above.
//
// Known limitation:
// 1. `torch.GC()` should be called in only one goroutine in a GoTorch program
//    (typically the main goroutine) exactly before the training/testing loop
//    starts. If we call `torch.GC()` in a unit test case, we have to call
//    `runtime.LockOSThread` manually because the `go test` cmd tool will start
//    new goroutines to run the cases.
// 2. `torch.FinishGC()` should only be called in the same goroutine as the
//    `torch.GC()` function exactly after a training/testing loop ends. Or the
//    goroutine may hang forever.
//
// In general, `torch.GC()` and `torch.FinishGC()` are low-level functions that
// ordinary users don't have to care about. GoTorch provides high-level APIs
// that wraps the two functions.

package torch_internal

// #cgo CPPFLAGS: -I/usr/local/include -I/usr/local/include/cgotorch
// #cgo LDFLAGS: -L/usr/local/lib -lc10 -ltorch_cpu -ltorch -lcgotorch
// #include "cgotorch/cgotorch.h"
// import "C"
// import (
//     "runtime"
//     "sync"
//     "unsafe"
// )

// var (
//     tensorFinalizersWG = &sync.WaitGroup{}
// )

// SetTensorFinalizer sets a finalizer to the tensor
// func SetTensorFinalizer(t *unsafe.Pointer) {
//     // We don't want the following conditional and the finalizer using
//     // different gcPrepared values, so we leverage p and closure here.
//     p := C.GCPrepared()
//     if p != 0 {
//         tensorFinalizersWG.Add(1)
//     }
//     runtime.SetFinalizer(t, func(ct *unsafe.Pointer) {
//         C.Torch_Tensor_Close(C.Tensor(*ct))
//         if p != 0 {
//             tensorFinalizersWG.Done()
//         }
//     })
// }

// // FinishGC should be called right after a train/predict loop
// func FinishGC() {
//     GC()
//     C.FinishGC()
// }

// // GC should be called at the beginning inside a train/predict loop
// func GC() {
//     runtime.GC()
//     if C.GCPrepared() == 0 {
//         C.PrepareGC()
//         return
//     }
//     tensorFinalizersWG.Wait()
// }
