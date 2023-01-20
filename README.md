<p align="center">
<img
    src="https://user-images.githubusercontent.com/2184469/211366276-5e1b5bf2-0ec6-48e8-9718-679992809c35.png"
    width="80%"
    alt="GoTorch"
/>
</p>

-----

GoTorch is a Golang front-end for [pytorch](https://github.com/pytorch/pytorch).

## Installation

Because GoTorch requires C libraries and some compilation, it cannot be
simply installed using `go get`. Instead, the recomended installation
pathway is to add gotorch as a git submodule in the `pkg` package of your
Go project, or otherwise check the files in directly. To add gotorch as
a submodule, from the top-level of your project, run:

```shell
git submodule add https://github.com/Kautenja/gotorch pkg/gotorch
```

Once gotorch has been cloned, it must be compiled using:

```shell
./pkg/gotorch/build.sh
```

This will download the libtorch binary for your system and compile the CGoTorch
wrapper for interfacing with CGO.

Golang must be told to use this local directory in-place of the remote code
by adding the following to your `go.mod`

```go.mod
replace github.com/Kautenja/gotorch v1.11.0 => ./pkg/gotorch
```

You may refer to the 
[example project](https://github.com/Kautenja/gotorch-example) for a working
demonstration of this approach.

## Documentation

Please refer to
[https://pkg.go.dev/github.com/Kautenja/gotorch](https://pkg.go.dev/github.com/Kautenja/gotorch)
for comprehensive documentation of the library.

## Development

##### Initialize

To initialize the go environment (e.g., download requirements, cleanup, etc.)

```shell
./main.sh init
```

##### Compile

To build the `libcgotorch` bridging library for `libtorch` and compile go code:

```shell
./main.sh build
```

##### Unit Test

To run test cases (after compiling code with `./main.sh build`):

```shell
./main.sh test
```

##### Continuous Integration

For convenience, the last three commands can be executed in series using:

```shell
./main.sh ci
```

## Docker Container

By default, development code is compiled directly for the host operating system
(assumed to be either MacOS or Linux.) Development functions can optionally be
executed within a Docker container using the `-d` flag. First, to build the
development Docker image:

```shell
./main.sh builddocker
```

For example, to run CI within the Docker container, use:

```shell
./main.sh -d ci
```
