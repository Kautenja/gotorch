// Go bindings for libtorch.
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

package torch

// #cgo CPPFLAGS: -I/usr/local/include -I/usr/local/include/cgotorch
// #cgo LDFLAGS: -L/usr/local/lib -lc10 -ltorch_cpu -ltorch -lcgotorch
// #include <stdio.h>
// #include <stdlib.h>
// #include "cgotorch/cgotorch.h"
import "C"
import (
	// "os"
	"runtime"
)

// Set the global gradient generation state.
func SetGradEnabled(value bool) { C.SetGradEnabled(C.bool(value)) }

// Return the global gradient generation state.
func IsGradEnabled() bool { return bool(C.IsGradEnabled()) }

// Set the random number generator seed.
func ManualSeed(seed int64) { C.ManualSeed(C.int64_t(seed)) }

// Set the number of threads used for intraop parallelism on CPU.
func SetNumThreads(numThreads int32) { C.SetNumThreads(C.int32_t(numThreads)) }

// TODO: set_default_dtype
// TODO: get_default_dtype
// TODO: set_default_tensor_type

func init() {
	// The goroutine scheduler has the following properties that might create
	// new OS threads for Cgo applications:
	// 1. The Go scheduler creates one or more P's, each represents a run queue,
	//    at the program startup time.
	// 2. The Go scheduler creates an OS thread M1 to run goroutines waiting in
	//    the run queue of a P.
	// 3. If a goroutine G makes a Cgo or system call, and the call takes a long
	//    time (>20ms), the Go scheduler would be afraid that it will go on
	//    occupying that thread for more time, so it creates a new OS thread M2
	//    for P to run other goroutines in the run queue.
	// 4. After the Cgo call completes, the goroutine G goes back to the run
	//    queue of the P. The next time it gets a turn to run, it runs on M2
	//    instead of reusing M1, because M2 is now the default thread of P.
	// 5. If G makes a new long-run Cgo call again, the Go scheduler would let G
	//    keeps using M2, and creates M3 for P to run other goroutines in its
	//    run queue.
	// In a nutshell, the above Go scheduler mechanism will create new threads
	// for a Cgo program like GoTorch and cause the main goroutine to migrate
	// from one OS thread to another.

	// The process may create several new threads. Moreover, libtorch and its
	// underlying libraries use thread local storage extensively, both for
	// caching and threads controlling, each thread that calls a `forward`
	// method may create its own computation threads and local storage. The
	// above Go mechanism may create so many threads that the RAM cannot afford,
	// because the Go runtime will never recycle OS threads at the moment. As a
	// result, a GoTorch program will occupy much RAM (the TLSs) and create many
	// threads (the computation threads). (This is an inherent problem of
	// libtorch, it also exists in C++. For a C++ example, See
	// https://github.com/wangkuiyi/gotorch/issues/331)
	// In order to alleviate the problem, we have to limit the default threads
	// number of OMP in GoTorch and lock the main goroutine to a fixed OS
	// thread to avoid migration.

	// Avoid creating too many threads: the original default setting of
	// OMP_NUM_THREADS (defaults to the core number in libtorch) may degrade
	// performance on GPUs because too many threads will increase the overhead
	// of context switching. See https://github.com/wangkuiyi/gotorch/issues/321
	// for details.
	// if os.Getenv("OMP_NUM_THREADS") == "" && os.Getenv("MKL_NUM_THREADS") == "" {
	//     SetNumThreads(int32(runtime.NumCPU()) / 2)
	// }

	// Prevent Cgo call from migrating to another system thread, hence the TLS
	// cache in libtorch would not take too much RAM.
	// See https://github.com/wangkuiyi/gotorch/issues/273 for details.
	runtime.LockOSThread()
}
