// GoTorch port of torchvision.ops.clip_boxes_to_image
//
// Copyright (c) 2023 Christian Kauten
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

package vision_ops

import (
    "fmt"
    "github.com/Kautenja/gotorch"
)

// Clip boxes so that they lie inside an image of size size. Boxes are
// specified by their (xmin, ymin, xmax, ymax) coordinates.
func ClipBoxesToImage(boxes torch.Tensor, height, width int64) torch.Tensor {
    shape := boxes.Shape()
    if len(shape) < 2 || shape[1] != 4 {
        panic(fmt.Sprintf("Expected inputs to be in (N, 4, ...) format, but received tensor with shape %v", shape))
    }
    dtype := boxes.Dtype()
    // Create the metadata tensors for the clamping operation.
    min := torch.NewTensor([]int64{0}).CastTo(dtype)
    xmax := torch.NewTensor([]int64{width}).CastTo(dtype)
    ymax := torch.NewTensor([]int64{height}).CastTo(dtype)
    // Select the x and y hyper columns (assumed to be interlaced) and clamp.
    x := boxes.Slice(1, 0, shape[1], 2).Clamp(min, xmax)
    y := boxes.Slice(1, 1, shape[1], 2).Clamp(min, ymax)
    // Stack the boxes back together and flatten them back to interlaced format.
    boxes = torch.Stack([]torch.Tensor{x, y}, 2).Flatten(1, 2)
    return boxes
}

// In-place version of ClipBoxesToImage.
func ClipBoxesToImage_(boxes torch.Tensor, height, width int64) torch.Tensor {
    shape := boxes.Shape()
    if len(shape) < 2 || shape[1] != 4 {
        panic(fmt.Sprintf("Expected inputs to be in (N, 4, ...) format, but received tensor with shape %v", shape))
    }
    dtype := boxes.Dtype()
    // Create the metadata tensors for the clamping operation.
    min := torch.NewTensor([]int64{0}).CastTo(dtype)
    xmax := torch.NewTensor([]int64{width}).CastTo(dtype)
    ymax := torch.NewTensor([]int64{height}).CastTo(dtype)
    // Select the x and y hyper columns (assumed to be interlaced) and clamp.
    boxes.Slice(1, 0, shape[1], 2).Clamp_(min, xmax)
    boxes.Slice(1, 1, shape[1], 2).Clamp_(min, ymax)
    return boxes
}
