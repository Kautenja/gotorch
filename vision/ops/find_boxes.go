// Methods for filtering bounding boxes based on size constraints.
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

// Find boxes within a particular inclusive minimum and maximum bound. Boxes
// are expected to be shaped as  (N, 4, ...) in (xmin, ymin, xmax, ymax) format.
func FindBoxesInSizeRange(
	boxes *torch.Tensor,
	widthMin, heightMin, widthMax, heightMax int64,
) *torch.Tensor {
	shape := boxes.Shape()
	if len(shape) < 2 || shape[1] != 4 {
		panic(fmt.Sprintf("Expected inputs to be in (N, 4, ...) format, but received tensor with shape %v", shape))
	}
	boxes = boxes.CastTo(torch.Float)
	// Calculate the width and height from the xmin,ymin,xmax,ymax boxes.
	width := boxes.Slice(1, 2, 3, 1).Sub(boxes.Slice(1, 0, 1, 1), 1.0)
	height := boxes.Slice(1, 3, 4, 1).Sub(boxes.Slice(1, 1, 2, 1), 1.0)
	// Calculate vectors determining if the boxes are within the bounds.
	isWideEnough := width.GreaterEqual(torch.FullLike(width, float32(widthMin)))
	isNotTooWide := width.LessEqual(torch.FullLike(width, float32(widthMax)))
	isTallEnough := height.GreaterEqual(torch.FullLike(height, float32(heightMin)))
	isNotTooTall := height.LessEqual(torch.FullLike(height, float32(heightMax)))
	// Aggregate the boolean vectors using logical AND operations.
	return isWideEnough.LogicalAnd(isNotTooWide).LogicalAnd(isTallEnough).LogicalAnd(isNotTooTall)
}

// Find boxes that are larger than or equal to a given size. Boxes are expected
// to be shaped as  (N, 4, ...) in (xmin, ymin, xmax, ymax) format.
func FindLargeBoxes(boxes *torch.Tensor, widthMin, heightMin int64) *torch.Tensor {
	shape := boxes.Shape()
	if len(shape) < 2 || shape[1] != 4 {
		panic(fmt.Sprintf("Expected inputs to be in (N, 4, ...) format, but received tensor with shape %v", shape))
	}
	boxes = boxes.CastTo(torch.Float)
	// Calculate the width and height from the xmin,ymin,xmax,ymax boxes.
	width := boxes.Slice(1, 2, 3, 1).Sub(boxes.Slice(1, 0, 1, 1), 1.0)
	height := boxes.Slice(1, 3, 4, 1).Sub(boxes.Slice(1, 1, 2, 1), 1.0)
	// Calculate vectors determining if the boxes are larger than the size.
	isWideEnough := width.GreaterEqual(torch.FullLike(width, float32(widthMin)))
	isTallEnough := height.GreaterEqual(torch.FullLike(height, float32(heightMin)))
	// Aggregate the boolean vectors using logical AND operations.
	return isWideEnough.LogicalAnd(isTallEnough)
}

// Find boxes that are smaller than or equal to a given size. Boxes are
// expected to be shaped as  (N, 4, ...) in (xmin, ymin, xmax, ymax) format.
func FindSmallBoxes(boxes *torch.Tensor, widthMax, heightMax int64) *torch.Tensor {
	shape := boxes.Shape()
	if len(shape) < 2 || shape[1] != 4 {
		panic(fmt.Sprintf("Expected inputs to be in (N, 4, ...) format, but received tensor with shape %v", shape))
	}
	boxes = boxes.CastTo(torch.Float)
	// Calculate the width and height from the xmin,ymin,xmax,ymax boxes.
	width := boxes.Slice(1, 2, 3, 1).Sub(boxes.Slice(1, 0, 1, 1), 1.0)
	height := boxes.Slice(1, 3, 4, 1).Sub(boxes.Slice(1, 1, 2, 1), 1.0)
	// Calculate vectors determining if the boxes are smaller than the size.
	isNotTooWide := width.LessEqual(torch.FullLike(width, float32(widthMax)))
	isNotTooTall := height.LessEqual(torch.FullLike(height, float32(heightMax)))
	// Aggregate the boolean vectors using logical AND operations.
	return isNotTooWide.LogicalAnd(isNotTooTall)
}
