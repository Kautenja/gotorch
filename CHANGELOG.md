# Change log

## v1.11.0-0.1.4

-   Resolve non-deterministic race condition between Go struct finalizers and
    calls to methods that access C resources. Update functions with
    `runtime.KeepAlive` as needed to prevent a segmentation fault from
    occurring due to early finalization of Go structs. I.e., this

    ```go
    func Foo(input *Tensor) (output *Tensor) {
        output = &Tensor{}
        C.SomeCFunction(&output.Pointer, input.Pointer)
        return
    }
    ```

    becomes this

    ```go
    func Foo(input *Tensor) (output *Tensor) {
        output = &Tensor{}
        C.SomeCFunction(&output.Pointer, input.Pointer)
        // Ensure the input is not finalized until SomeCFunction returns
        runtime.KeepAlive(input)
        return
    }
    ```

## v1.11.0-0.1.3

Refactor usage of `SetFinalizer` following this
[post](https://groups.google.com/g/golang-dev/c/DMiUkpS1uyQ/m/0sHpTacG00YJ)
from this
[thread](https://groups.google.com/g/golang-dev/c/DMiUkpS1uyQ?pli=1).

> I think Dmitriy's picture is gloomier than the reality. Yes, there are
> corner cases in which finalizers do not run, but most will not come up in
> real usage. The only remotely likely problem setting a finalizer on a
> complex structure that you didn't realize had a cycle. Instead, set the
> finalizer on a simple structure. Make sure the structure you are setting the
> finalizer on does not have a pointer to itself (even indirectly) and you
> will be fine. For example, if you defined:

```golang
type cpointer struct {
    p unsafe.Pointer
}

func (cp *cpointer) free() {
    C.free(cp.p)
    cp.p = nil
}

func newCPointer(p unsafe.Pointer) *cpointer {
    cp := &cpointer{p}
    runtime.SetFinalizer(cp, (*cpointer).free)
    return cp
}
```

This enables better integration with the garbage collector and resolved the
bug where devices were being finalized before usage in a downstream function.
I.e., certain cases of the following semantics (not this precise syntax) would
result in de-allocation of the device before usage within the following
`CopyTo` function.

```golang
device := torch.NewDevice("cpu")
aTensor.CopyTo(device)
```

This change affects all structures with C bindings, i.e.,

-   `torch.Device`
-   `torch.Tensor`
-   `torch.TensorOptions`
-   `jit.JitModule`
-   `torch.IValue`

## v1.11.0-0.1.2

-   Fix a memory leak in the `torch.Decode` function.
-   Update installation and development scripts to terminate if any
    sub-commands fail
    -   This resolves issues of silent installation failures that were being
        observed when deploying the code into Docker containers

## v1.11.0-0.1.1

-   Convert spaces to tabs following Go standard conventions

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
