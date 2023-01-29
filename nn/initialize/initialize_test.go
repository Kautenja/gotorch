// test cases for initialize.go
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

package nn_initialize_test

import (
    // "testing"
    // "github.com/stretchr/testify/assert"
    // "github.com/Kautenja/gotorch"
    // initialize "github.com/Kautenja/gotorch/nn/initialize"
)

// func forAllElements(t *testing.T, x torch.Tensor, ck func(elem interface{}) bool) {
//     shape := x.Shape()
//     idx := make([]int64, len(shape))

//     increment := func(idx, shape []int64) bool {
//         i := 0
//         for {
//             idx[i]++
//             if idx[i] >= shape[i] {
//                 if i+1 >= len(shape) {
//                     return false // no increment any more
//                 }
//                 idx[i] = 0
//                 i++
//             } else {
//                 break
//             }
//         }
//         return true // successfully increased idx
//     }

//     for {
//         if !ck(x.Index(idx...).Item()) {
//             t.Fatalf("forAllElements failed at index %v", idx)
//         }
//         if !increment(idx, shape) {
//             break
//         }
//     }
// }

// func TestNormal(t *testing.T) {
//     var x torch.Tensor
//     assert.Panics(t, func() { initialize.Normal(&x, 0.1, 0.2) })

//     x = torch.Empty([]int64{2, 3}, torch.NewTensorOptions())
//     initialize.Normal(&x, 0.1, 0.2)
//     assert.NotNil(t, x.T)
// }

// func TestUniform(t *testing.T) {
//     var x torch.Tensor
//     assert.Panics(t, func() { initialize.Uniform(&x, 11.1, 22.2) })

//     x = torch.Empty([]int64{2, 3}, torch.NewTensorOptions())
//     initialize.Uniform(&x, 11, 22)
//     forAllElements(t, x, func(elem interface{}) bool {
//         return float32(11.1) <= elem.(float32)
//     })
//     forAllElements(t, x, func(elem interface{}) bool {
//         return elem.(float32) < float32(22.2)
//     })
// }

// func TestZeros(t *testing.T) {
//     var x torch.Tensor
//     assert.Panics(t, func() { initialize.Zeros(&x) })

//     x = torch.Empty([]int64{2, 3}, torch.NewTensorOptions())
//     initialize.Zeros(&x)
//     forAllElements(t, x, func(elem interface{}) bool {
//         return float32(0) == elem.(float32)
//     })
// }

// func TestOnes(t *testing.T) {
//     var x torch.Tensor
//     assert.Panics(t, func() { initialize.Ones(&x) })

//     x = torch.Empty([]int64{2, 3}, torch.NewTensorOptions())
//     initialize.Ones(&x)
//     forAllElements(t, x, func(elem interface{}) bool {
//         return float32(1) == elem.(float32)
//     })
// }
