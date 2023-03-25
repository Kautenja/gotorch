// GoTorch port of functional elements of albumentations.PadIfNeeded.
//
// Copyright (c) 2023 Christian Kauten
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

// Pad the input tensor to the given minimum height and width. If padding is
// necessary, use the given padding mode. When the padding mode is constant,
// use the given constant padding value. Returns the padded tensor and the
// padding that was applied in [left, right, top, bottom] format.
func PadIfNeeded(
	tensor *torch.Tensor,
	min_height, min_width int64,
	mode F.PadMode,
	value ...float64,
) (*torch.Tensor, []int64) {
	shape := tensor.Shape()
	dim := len(shape)
	if dim < 2 { panic("PadIfNeeded requires tensor with 2 or more dimensions") }
	H := math.Max(0.0, (float64(min_height) - float64(shape[int64(dim - 2)])) / 2)
	top := int64(math.Floor(H))
	bottom := int64(math.Ceil(H))
	W := math.Max(0.0, (float64(min_width) - float64(shape[int64(dim - 1)])) / 2)
	left := int64(math.Floor(W))
	right := int64(math.Ceil(W))
	padding := []int64{left, right, top, bottom}
	// Padding vectors are constructed from the last dimension backwards. This
	// is convenient for vision where the format is NCHW typically. I.e., we
	// can always create a length 4 slice in [left, right, top, bottom] format.
	return F.Pad(tensor, padding, mode, value...), padding
}
