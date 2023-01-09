// GoTorch port of albumentations.PadIfNeeded.
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

package vision_transforms

import (
    "github.com/Kautenja/gotorch"
    F "github.com/Kautenja/gotorch/nn/functional"
    T "github.com/Kautenja/gotorch/vision/transforms/functional"
)

// A transformer that pads images to a minimum size.
type PadIfNeededTransformer struct {
    // The minimum height to pad images to.
    min_height int64
    // The minimum width to pad images to.
    min_width int64
    // The mode to use when padding.
    mode F.PadMode
    // An optional constant value for constant padding modes.
    value []float64
}

// Create a new PadIfNeededTransformer with given parameters.
func PadIfNeeded(min_height, min_width int64, mode F.PadMode, value ...float64) *PadIfNeededTransformer {
    if min_height <= 0 { panic("min_height should be greater than 0") }
    if min_width <= 0 { panic("min_width should be greater than 0") }
    return &PadIfNeededTransformer{min_height, min_width, mode, value}
}

// Forward pass an image through the transformer to pad it if needed.
func (t PadIfNeededTransformer) Forward(tensor torch.Tensor) torch.Tensor {
    outputs, _ := T.PadIfNeeded(tensor, t.min_height, t.min_width, t.mode, t.value...)
    return outputs
}
