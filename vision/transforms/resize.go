// GoTorch port of torchvision.transforms.Resize
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

package vision_transforms

import (
	"github.com/Kautenja/gotorch"
	F "github.com/Kautenja/gotorch/nn/functional"
)

// A transformer that resizes images to a certain size at the center
type ResizeTransformer struct {
	// The height and width to resize images to
	height, width int64
	// The kind of interpolation to use when resizing images
	interpolation F.InterpolateMode
	// Whether to align corners
	alignCorners bool
	// Whether to use anti-aliasing filters
	antialias bool
}

// Create a new ResizeTransformer with given parameters.
func Resize(height, width int64, interpolation F.InterpolateMode, alignCorners, antialias bool) *ResizeTransformer {
	if height <= 0 { panic("height should be greater than 0") }
	if width <= 0 { panic("width should be greater than 0") }
	return &ResizeTransformer{height, width, interpolation, alignCorners, antialias}
}

// Forward pass an image through the transformer to resize it.
func (t ResizeTransformer) Forward(tensor *torch.Tensor) *torch.Tensor {
	return F.InterpolateSize(tensor, []int64{t.height, t.width}, t.interpolation, t.alignCorners, t.antialias)
}
