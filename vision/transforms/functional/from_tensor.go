// GoTorch port of torchvision.transforms.functional.from_tensor.
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
    "unsafe"
    "fmt"
    "image"
    "github.com/Kautenja/gotorch"
)

// Convert `torch.Tensor` to `image.Image`.
func FromTensor(tensor torch.Tensor) image.Image {
    // Images are expected in HWC format.
    shape := tensor.Shape()
    if len(shape) != 3 {
        panic(fmt.Sprintf("Expected tensor to be 3-dimensional but tensor has shape %v", shape))
    }
    // Images can be gray, RGB, or RGBA only.
    channels := shape[0]
    height := shape[1]
    width := shape[2]
    if channels == 1 {
        // Create a mock alpha channel using "like" semantics to convey type.
        alpha := torch.OnesLike(tensor.Slice(0, 0, 1, 1))
        // 1HW -> 4HW
        tensor = torch.Cat([]torch.Tensor{tensor, tensor, tensor, alpha}, 0)
    } else if channels == 3 {
        // Create a mock alpha channel using "like" semantics to convey type.
        alpha := torch.OnesLike(tensor.Slice(0, 0, 1, 1))
        // 3HW -> 4HW
        tensor = torch.Cat([]torch.Tensor{tensor, alpha}, 0)
    } else if channels != 4 {
        panic(fmt.Sprintf("Expected tensor to have 1, 3, or 4 channels, but found %v", channels))
    }
    // [0., 1.] -> [0., 255.]; type -> byte; CHW -> HWC
    tensor = tensor.Mul(torch.FullLike(tensor, 255.0)).CastTo(torch.Byte).Permute(1, 2, 0)
    // image.Image are assumed to be in HWC format, this is the expected pixel
    // layout for RGBA image buffers in the GoLang image processing library.
    frame := image.NewRGBA(image.Rect(0, 0, int(width), int(height)))
    target := torch.TensorFromBlob(unsafe.Pointer(
        &frame.Pix[0]), torch.Byte, []int64{height, width, 4})
    target.Copy_(tensor)
    return frame
}
