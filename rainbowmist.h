#ifndef RAINBOWMIST_H_
#define RAINBOWMIST_H_

//
// RaibowMist, simple header only CUDA/OpenCL/C++11 single kernel description utility.
//

/*
The MIT License (MIT)

Copyright (c) 2017 - present Light Transport Entertainment, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifdef __CUDACC__

#define RAINBOWMIST_CUDA (1)

#define RAINBOWMIST_USE_GLM (0)
#if RAINBOWMIST_USE_GLM

// Use glm for vector swizzle.
#ifndef GLM_FORCE_SWIZZLE
#define GLM_FORCE_SWIZZLE
#endif

#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif

#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/geometric.hpp"

using namespace glm;

// From CUDA SDK samples
#include "helper_math.h"

#define vnormalize(x) normalize(x)
#define vcross(a, b) cross(a, b)
#define vdot(a, b) dot(a, b)

#else // !RAINBOWMIST_USE_GLM

// Include helper_math firstly
#include "helper_math.h"

typedef float2 vec2;
typedef float3 vec3;
typedef float4 vec4;

#define vnormalize(x) normalize(x)
#define vcross(a, b) cross(a, b)
#define vdot(a, b) dot(a, b)

#endif // RAINBOWMIST_USE_GLM

//#include <float.h> // import FLT_EPSILON

#ifndef FLT_EPSILON
#define FLT_EPSILON (0x1.0p-23f)
#endif

#define DEVICE __device__
#define HOST   __host__
#define GLOBAL 
#define LOCAL 
#define KERNEL extern "C" __global__


DEVICE static inline vec2 make_vec2(float a, float b)
{
  vec2 ret;
  ret.x = a;
  ret.y = b;
  return ret;
}

DEVICE static inline vec3 make_vec3(float a, float b, float c)
{
  vec3 ret;
  ret.x = a;
  ret.y = b;
  ret.z = c;
  return ret;
}

DEVICE static inline vec4 make_vec4(float a, float b, float c, float d)
{
  vec4 ret;
  ret.x = a;
  ret.y = b;
  ret.z = c;
  ret.w = d;
  return ret;
}

#elif defined(OPENCL)  // NOTE(LTE): Application must pass `-D OPENCL` as a OpenCL compile flags.

// We only support OpenCL 1.2

#define RAINBOWMIST_OPENCL (1)

#define DEVICE 
#define KERNEL __kernel
#define GLOBAL __global
#define LOCAL  __local

#define vnormalize(x) normalize(x)
#define vcross(a, b) cross(a, b)
#define vdot(a, b) dot(a, b)

#define sinf(x) sin(x)
#define cosf(x) cos(x)
#define sqrtf(x) sqrt(x)
#define fabsf(x) fabs(x)

typedef float2 vec2;
typedef float3 vec3;
typedef float4 vec4;

#if !defined(__APPLE__)
static inline vec2 make_vec2(float a, float b)
{
  return (vec2)(a, b);
}

static inline vec3 make_vec3(float a, float b, float c)
{
  return (float3)(a, b, c);
}

static inline vec4 make_vec4(float a, float b, float c, float d)
{
  return (float4)(a, b, c, d);
}
#endif

#else // C++

#define RAINBOWMIST_CPP11 (1)

#include <cfloat>
#include <cmath>

#ifndef RAINBOWMIST_USE_GLM
#define RAINBOWMIST_USE_GLM (1)
#endif

#ifdef RAINBOWMIST_USE_GLM

#ifndef GLM_FORCE_SWIZZLE
#define GLM_FORCE_SWIZZLE
#endif

#include "glm/glm.hpp"
#include "glm/geometric.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wheader-hygiene"
#endif

using namespace glm;

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#else // !RAINBOWMIST_USE_GLM

// Use CxxSwizzle.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-qualifiers"
#endif

#include "swizzle/glsl/scalar_support.h"
#include "swizzle/glsl/vector.h"
#include "swizzle/glsl/vector_functions.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

typedef swizzle::glsl::vector<float, 2> vec2;
typedef swizzle::glsl::vector<float, 3> vec3;
typedef swizzle::glsl::vector<float, 4> vec4;

// ----------------------------------------------------
#endif // RAINBOWMIST_USE_GLM

// TODO(LTE): Implement more stuff.
#define vnormalize(x) normalize(x)
#define vcross(a, b) cross(a, b)
#define vdot(a, b) dot(a, b)

// NOTE(LTE): Liminatation
// - No shared variable support

#define KERNEL static
#define DEVICE 
#define GLOBAL 
#define SHARED ERROR


static inline vec2 make_vec2(float a, float b)
{
  vec2 ret;
  ret.x = a;
  ret.y = b;
  return ret;
}

static inline vec3 make_vec3(float a, float b, float c)
{
  vec3 ret;
  ret.x = a;
  ret.y = b;
  ret.z = c;
  return ret;
}

static inline vec4 make_vec4(float a, float b, float c, float d)
{
  vec4 ret;
  ret.x = a;
  ret.y = b;
  ret.z = c;
  ret.w = d;
  return ret;
}

#endif 

#endif // RAINBOWMIST_H_
