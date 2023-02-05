# Change log

## v1.11.0-0.1.0

-   vision
    -   Change semantics of `vision/transforms/functional` `Crop` to follow
        that of python's torchvision
    -   Implement new `vision/transforms/functional` `SafeCrop` that it
        implements the `0.0.0` logic of `Crop`, i.e., where bounding boxes are
        clipped to the safe area
-   torch
    -   Update `torch.TorchVersion()` to access the version of libtorch
        directly from the C++ back-end, opposed to a hard-coded string constant

## v1.11.0-0.0.0

-   Support for libtorch 1.11.0
