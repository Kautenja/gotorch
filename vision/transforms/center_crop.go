// GoTorch port of torchvision.transforms.CenterCrop.
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
)

// A transformer that crops images to a certain size at the center
type CenterCropTransformer struct {
    height, width int64
}

// Create a new CenterCropTransformer with given height and width.
func CenterCrop(height, width int64) *CenterCropTransformer {
    if height <= 0 { panic("height should be greater than 0") }
    if width <= 0 { panic("width should be greater than 0") }
    return &CenterCropTransformer{height, width}
}

// Forward pass an image through the transformer to center crop it.
func (t CenterCropTransformer) Forward(tensor torch.Tensor) torch.Tensor {
    shape := tensor.Shape()
    if len(shape) < 2 {
        panic("CenterCrop only supports tensors with 2 or more dimensions")
    }
    // Determine which dimensions the height and width are in
    height_dim := int64(len(shape) - 2)
    width_dim := int64(len(shape) - 1)
    // Select the height and width from the shape slice.
    height := shape[height_dim]
    width := shape[width_dim]
    // Calculate the offset for the height and width.
    offset_height := int64((height - t.height) / 2)
    offset_width := int64((width - t.width) / 2)
    tensor = tensor.Slice(height_dim, offset_height, height - offset_height, 1)
    tensor = tensor.Slice(width_dim, offset_width,  width - offset_width,   1)
    return tensor
}
