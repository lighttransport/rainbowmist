# RainbowMist 🌈🌫️, header only OpenCL/CUDA/C++11 single compute kernel description utility.

RainbowMist 🌈🌫️ is a simple macro and template based single compute kernel.

## Status

**Experimenal**.

## Supported OSes

* Winwows 7 64bit or later(32bit not supported). 
* Ubuntu 16.04 or later(32bit not supported). 
* macOS 10.12. 

## Supported platforms

* CUDA 8.0(7.0 or 7.5 may work but not tested)
* OpenCL 1.2
* C++11 compiler

## Advantages

* Easy to maintain your compute kenerl code among CUDA/OpenCL/C++11.
* Easy to adapt to your application.
* (Rathe) Easy to debug in source code level

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
$ git submodule update --init
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

## TODO

* [ ] OpenGL GLSL compute shader?

## Third party licenses

* helper_math.h : Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
* glm : Happy Bunny License (Modified MIT) or the MIT License
* Catch : 
* EasyCL
* clew
* cuew

