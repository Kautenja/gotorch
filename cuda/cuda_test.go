// test cases for cuda.go
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

package torch_cuda_test

import (
    "testing"
    "github.com/stretchr/testify/assert"
    // torch "github.com/Kautenja/gotorch"
    cuda "github.com/Kautenja/gotorch/cuda"
)

func TestCUDAIsAvailable(t *testing.T) {
    assert.NotPanics(t, func() {
        if cuda.IsAvailable() {
            t.Log("CUDA is available")
        } else {
            t.Log("No CUDA found")
        }
    })
}

func TestCUDAIsCUDNNAvailable(t *testing.T) {
    assert.NotPanics(t, func() {
        if cuda.IsCUDNNAvailable() {
            t.Log("CUDNN is available")
        } else {
            t.Log("No CUDNN found")
        }
    })
}

// func getDefaultDevice() torch.Device {
//  var device torch.Device
//  if cuda.IsAvailable() {
//      device = torch.NewDevice("cuda")
//  } else {
//      device = torch.NewDevice("cpu")
//  }
//  return device
// }
// func TestCUDAStreamPanics(t *testing.T) {
//  a := assert.New(t)
//  device := getDefaultDevice()
//  if cuda.IsAvailable() {
//      a.NotPanics(func() {
//          cuda.GetCurrentCUDAStream(device)
//      })
//  } else {
//      a.Panics(func() {
//          cuda.GetCurrentCUDAStream(device)
//      })
//      a.Panics(func() {
//          cuda.NewCUDAStream(device)
//      })
//  }
// }

// func TestMultiCUDAStream(t *testing.T) {
//  if !cuda.IsAvailable() {
//      t.Skip("skip TestMultiCUDAStream which only run on CUDA device")
//  }
//  a := assert.New(t)
//  device := getDefaultDevice()
//  currStream := cuda.GetCurrentCUDAStream(device)
//  defer cuda.SetCurrentCUDAStream(currStream)
//  // create a new CUDA stream
//  stream := cuda.NewCUDAStream(device)
//  // switch to the new CUDA stream
//  cuda.SetCurrentCUDAStream(stream)
//  // copy Tensor from host to device async
//  input := torch.RandN([]int64{100, 200}, true).PinMemory()
//  input.CUDA(device, true /**nonBlocking=true**/)
//  // wait until all tasks completed
//  stream.Synchronize()
//  // make sure all tasks completed
//  a.True(stream.Query())
// }
