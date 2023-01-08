// GoTorch port of functional elements of albumentations.LongestMaxSize.
//
// Copyright (c) 2022 Christian Kauten
// Copyright (c) 2022 Sensory, Inc.
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

package vision_transforms_functional

import (
    "math"
    "github.com/Kautenja/gotorch"
    F "github.com/Kautenja/gotorch/nn/functional"
)

// Resize a tensor to have the longest size equal to `size`. If interpolation
// is required, e.g., if the tensor has max size less than or equal to `size`,
// use the given interpolation mode, corner alignment setting, and optional
// anti-aliasing.
func LongestMaxSize(
    tensor torch.Tensor,
    size int64,
    interpolation F.InterpolateMode,
    alignCorners, antialias bool,
) torch.Tensor {
    shape := tensor.Shape()
    dim := len(shape)
    if dim < 3 { panic("LongestMaxSize requires tensor with 3 or more dimensions") }
    H := float64(shape[int64(dim - 2)])
    W := float64(shape[int64(dim - 1)])
    scale := float64(size) / math.Max(H, W)
    if scale != 1.0 {
        Htarget := int64(scale * H)
        Wtarget := int64(scale * W)
        return F.InterpolateSize(tensor, []int64{Htarget, Wtarget}, interpolation, alignCorners, antialias)
    }
    return tensor
}
