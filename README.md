# RainbowMist 🌈🌫️, header only OpenCL/CUDA/C++11 single compute kernel descripion utility.

RainbowMist 🌈🌫️ is a simple C++ macro and template based single compute kernel utility.

Write and debug your kernel in C++11, then run it on GPU(OpenCL/CUDA).

## Status

**Experimenal**.

## Supported OSes

* [x] Winwows 10 64bit or later(32bit not supported). 
* [x] Ubuntu 16.04 or later(32bit not supported). 
* [ ] macOS 10.12 or later 

## Supported platforms

* CUDA 8.0(7.0 or 7.5 may work but not tested)
* OpenCL 1.2

## Requirements

* C++11 compiler

RainbowMist uses cuew(CUDA wrangler wrapper)/nvrtc and clew(OpenCL wrangler wrapper) and does not require neither CUDA SDK nor OpenCL SDK when compiling.

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

`not` operator is not available in C++11(since `not` is a reserved keyword).

## TODO

* [ ] OpenGL GLSL compute shader?
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
* clew :
* cuew : Modified Apache 2.0 License.

