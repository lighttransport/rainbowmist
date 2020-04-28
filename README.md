# RainbowMist 🌈🌫️, header only OpenCL/CUDA/C++11 unified compute kernel descripion utility.

RainbowMist 🌈🌫️ is a simple C++ macro and template based unified compute kernel utility.

Write and debug your kernel in C++11, then run it on GPU(OpenCL/CUDA).

## Status

**Experimenal**.

## Supported OSes

* [x] Winwows 10 64bit or later(32bit not supported). 
* [x] Ubuntu 16.04 or later(32bit not supported). 
* [ ] macOS 10.12 or later 

## Supported platforms

* CUDA 8.0 or later
  * 7.0 or 7.5 may work but not tested
  * CUDA 9.2 or later recommended
* OpenCL 1.2

## Requirements

* C++11 compiler

RainbowMist uses cuew(CUDA wrangler wrapper)/nvrtc and clew(OpenCL wrangler wrapper) and does not require neither CUDA SDK nor OpenCL SDK when compiling.

## How it works

### CUDA backend

RainbowMist runs a kernel using CLCudaAPI and cuew to run it on CUDA device.
For CUDA environment, it uses NVRTC(Runtime compilation) feature to compile a kernel at the runtime. NVRTC was introduced from CUDA 7.0, so it should be veryu mature in these days(CUDA 9.x or 10.x is a mainstream version as of 2020 April)

### OpenCL backend

RainbowMist runs a kernel using CLCudaAPI or EasyCL and clew to run it on OpenCL device.

## Advantages

* Easy to maintain your compute kenerl code among CUDA/OpenCL/C++11.
* Easy to adapt to your application.
* (Rather) Easy to debug in source code level

## Disadvantage

* Cannot use language specific features.

## Useful defines

```
RAINBOWMIST_CUDA    : defined when compiling code with CUDA
RAINBOWMIST_OPENCL  : defined when compiling code with OpenCL
RAINBOWMIST_CPP11   : defined when compiling code with C++11
```


## Setup

```
$ git submodule update --init --depth 1
```

## Running tests

Requires CMake 3.1 or later.


```
$ cd tests
$ cmake -Bbuild -H.
$ cd build
$ cmake --build .
```

### Windows

Tested on Visual Studio 2015 or later.

```
$ cd tests
$ cmake -Bbuild -H. -G "Visual Studio 14 2015 Win64"
```

## Limitation

`not` operator is not available in C++11 backend(since `not` is a reserved keyword in C++).

### Data alignment

You must be careful about the byte alignment and byte size of vector data type(e.g. `vec3`).
For OpenCL backend, `vec3` will be aligned to 16 bytes and have 16 bytes of its data.
C++11 and CUDA has 12 bytes of its data and may be aligned to 4, 8, 12 or 16 bytes.

Thus to use same data layout among C++11/CUDA/OpenCL, you'll need to add extra pad variables for C++11 and CUDA.
See [](tests/alignment.kernel) for details.

## TODO

* [ ] Write more examples.
* [ ] OpenGL/Vulkan compute shader backend?
* [ ] Implement more builtin functions, math functions.
* [x] Swizzle without glm.
  * CxxSwizle
* [ ] Common API to call Kernel function.

## Third party licenses

* helper_math.h : Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
* CxxSwizzle: MIT license. https://github.com/gwiazdorrr/CxxSwizzle
* glm : Happy Bunny License (Modified MIT) or the MIT License
* Catch : Boost license.
* EasyCL : Copyright (c) Hugh Perkins 2013, hughperkins at gmail. MPL license.
* clew : MIT license. https://github.com/OpenCLWrangler/clew
* cuew(CUDA 10.2 ready) : Modified Apache 2.0 License. https://github.com/syoyo/cuew

