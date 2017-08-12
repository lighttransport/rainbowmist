# RainbowMist, header only OpenCL/CUDA/C++11 single compute kernel description utility.

RainbowMist is a simple macro and template based single compute kernel.

## Status

Experimenal.

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

## Third party licenses

* helper_math.h : Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
* glm : Happy Bunny License (Modified MIT) or the MIT License
* Catch : 
* EasyCL
* clew
* cuew

