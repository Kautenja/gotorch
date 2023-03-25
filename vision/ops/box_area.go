// GoTorch port of torchvision.ops.box_area
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

// Compute the area of a set of bounding boxes. boxes are expected to be in
// (N, 4, ...) (xmin, ymin, xmax, ymax) format with 0 <= xmin < xmax and
// 0 <= ymin < ymax.
func BoxArea(boxes *torch.Tensor) *torch.Tensor {
	shape := boxes.Shape()
	if len(shape) < 2 || shape[1] != 4 {
		panic(fmt.Sprintf("Expected inputs to be in (N, 4, ...) format, but received tensor with shape %v", shape))
	}
	// Before selecting hyper-columns, cast the boxes to a floating point type.
	boxes = boxes.CastTo(torch.Float)
	// Select the individual component hyper-columns.
	xmin := boxes.Slice(1, 0, 1, 1)
	ymin := boxes.Slice(1, 1, 2, 1)
	xmax := boxes.Slice(1, 2, 3, 1)
	ymax := boxes.Slice(1, 3, 4, 1)
	// Calculate the height and width (in-place since we copied boxes already.)
	height := ymax.Sub_(ymin, 1.0)
	width := xmax.Sub_(xmin, 1.0)
	// Calculate the area (in-place since we copied stuff already.)
	area := height.Mul_(width)
	return area
}
