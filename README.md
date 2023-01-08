# GoTorch

A Golang front-end for
[libtorch](https://pytorch.org/cppdocs/api/library_root.html).

## Installation

Because GoTorch requires C libraries and some compilation, it cannot be
simply installed using `go get`. Instead, the recomended installation
pathway is to add gotorch as a git submodule in the `pkg` package of your
Go project, or otherwise check the files in directly, i.e., from the 
top-level of your project, run:

```shell
git submodule add https://github.com/Kautenja/gotorch pkg/gotorch
```

Once gotorch has been cloned, it must be compiled using:

```shell
./pkg/gotorch/build.sh
```

This will download the libtorch binary for your system and compile the CGoTorch
wrapper for interfacing with CGO. One may refer to the 
[example](https://github.com/Kautenja/gotorch-example) project for a working
demonstration of this approach.

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
