#ifndef RAINBOWMIST_H_
#define RAINBOWMIST_H_

//
// RaibowMist, simple header only CUDA/OpenCL/C++11 single kernel description
// utility.
//

/*
The MIT License (MIT)

Copyright (c) 2017 - 2020 Light Transport Entertainment, Inc.

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

#include "glm/geometric.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

using namespace glm;

// From CUDA SDK samples
#include "helper_math.h"

#define vnormalize(x) normalize(x)
#define vcross(a, b) cross(a, b)
#define vdot(a, b) dot(a, b)

#else  // !RAINBOWMIST_USE_GLM

// Include helper_math firstly
#include "helper_math.h"

typedef float2 vec2;
typedef float3 vec3;
typedef float4 vec4;
typedef uint2  uvec2;
typedef uint3  uvec3;
typedef uint4  uvec4;

#define vnormalize(x) normalize(x)
#define vcross(a, b) cross(a, b)
#define vdot(a, b) dot(a, b)
#define vmix(x, y, a) mix(x, y, a)

#endif  // RAINBOWMIST_USE_GLM

//#include <float.h> // import FLT_EPSILON

#ifndef FLT_EPSILON
#define FLT_EPSILON (1.19209e-07)
#endif

// Function modifiers
#define RM_DEVICE __device__
#define RM_KERNEL extern "C" __global__
#define RM_HOST __host__
// Variable modifiers
#define RM_GLOBAL
#define RM_LOCAL __shared__
#define RM_CONST __constant__
#define RM_PRIVATE

#define RM_STATIC_CAST(t, x) (t)(x)

RM_DEVICE static inline uvec3 GlobalId() {
  return make_uint3(blockDim.x * blockIdx.x + threadIdx.x,
               blockDim.y * blockIdx.y + threadIdx.y,
               blockDim.z * blockIdx.z + threadIdx.z);
}

RM_DEVICE static inline vec2 mix(vec2 x, vec2 y, float a) {
  return x + (y - x) * a;
}

RM_DEVICE static inline vec2 mix(vec2 x, vec2 y, vec2 a) {
  return x + (y - x) * a;
}

RM_DEVICE static inline vec2 make_vec2(float a, float b) {
  vec2 ret;
  ret.x = a;
  ret.y = b;
  return ret;
}

RM_DEVICE static inline vec3 make_vec3(float a, float b, float c) {
  vec3 ret;
  ret.x = a;
  ret.y = b;
  ret.z = c;
  return ret;
}

RM_DEVICE static inline vec4 make_vec4(float a, float b, float c, float d) {
  vec4 ret;
  ret.x = a;
  ret.y = b;
  ret.z = c;
  ret.w = d;
  return ret;
}

// NOTE(LTE): Application should pass `-DOPENCL` or `-DRAINBOWMIST_OPENCL`(macOS)
// as a OpenCL compilation flags.
#elif defined(OPENCL) || defined(RAINBOWMIST_OPENCL)

// We only support OpenCL 1.2

#ifndef RAINBOWMIST_OPENCL
#define RAINBOWMIST_OPENCL (1)
#endif

#define nullptr (0)

// Function modifiers
#define RM_DEVICE
#define RM_KERNEL __kernel
#define RM_HOST __host__
// Variable modifiers
#define RM_GLOBAL __global
#define RM_LOCAL __local
#define RM_CONST __constant
#define RM_PRIVATE __private

#define RM_STATIC_CAST(t, x) (t)(x)

typedef float2 vec2;
typedef float3 vec3;
typedef float4 vec4;
typedef uint2 uvec2;
typedef uint3 uvec3;
typedef uint4 uvec4;

static inline uvec3 GlobalId() {
  return (uvec3)(get_global_id(0), get_global_id(1), get_global_id(2));
}

#define vnormalize(x) normalize(x)
#define vcross(a, b) cross(a, b)
#define vdot(a, b) dot(a, b)
#define vmix(x, y, a) mix(x, y, a)

#define sinf(x) sin(x)
#define cosf(x) cos(x)
#define sqrtf(x) sqrt(x)
#define fabsf(x) fabs(x)

//#if !defined(__APPLE__)
static inline vec2 make_vec2(float a, float b) { return (vec2)(a, b); }

static inline vec3 make_vec3(float a, float b, float c) {
  return (float3)(a, b, c);
}

static inline vec4 make_vec4(float a, float b, float c, float d) {
  return (float4)(a, b, c, d);
}
//#endif

#else  // C++

#define RAINBOWMIST_CPP11 (1)

#include <cfloat>
#include <cmath>
#include <cstdio>

#ifndef RAINBOWMIST_USE_GLM
#define RAINBOWMIST_USE_GLM (0)
#endif

#if RAINBOWMIST_USE_GLM

#ifndef GLM_FORCE_SWIZZLE
#define GLM_FORCE_SWIZZLE
#endif

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4201)
#endif
#include "glm/geometric.hpp"
#include "glm/glm.hpp"

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wheader-hygiene"
#endif

using namespace glm;

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#else  // !RAINBOWMIST_USE_GLM

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
typedef swizzle::glsl::vector<int, 2> ivec2;
typedef swizzle::glsl::vector<int, 3> ivec3;
typedef swizzle::glsl::vector<int, 4> ivec4;
typedef swizzle::glsl::vector<unsigned int, 2> uvec2;
typedef swizzle::glsl::vector<unsigned int, 3> uvec3;
typedef swizzle::glsl::vector<unsigned int, 4> uvec4;

// ----------------------------------------------------
#endif  // RAINBOWMIST_USE_GLM

// TODO(LTE): Implement more stuff.
#define vnormalize(x) normalize(x)
#define vcross(a, b) cross(a, b)
#define vdot(a, b) dot(a, b)

#define vmix(x, y, a) mix(x, y, a)

// Function modifiers
#define RM_DEVICE
#define RM_KERNEL
#define RM_HOST
// Variable modifiers
#define RM_GLOBAL
#define RM_LOCAL
#define RM_CONST const
#define RM_PRIVATE

#define RM_STATIC_CAST(t, x) static_cast<t>(x)

// Global id mechanism for C++11 mode.
// In c++ mode, a kernel function must be called in a loop.
#include <atomic>
static std::atomic<unsigned int> rainbowmist_gobal_id;
static unsigned int rainbowmist_global_x_size = 0;
static unsigned int rainbowmist_global_y_size = 0;
static unsigned int rainbowmist_global_z_size = 0;

static void SetupGlobalId(unsigned int xs, unsigned int ys = 1,
                          unsigned int zs = 1) {
  rainbowmist_gobal_id = 0;
  rainbowmist_global_x_size = xs;
  rainbowmist_global_y_size = ys;
  rainbowmist_global_z_size = zs;
}

static inline uvec3 GlobalId() {
  unsigned int id = rainbowmist_gobal_id++;
  unsigned int x = id % rainbowmist_global_x_size;
  unsigned int y = (id / rainbowmist_global_x_size) % rainbowmist_global_y_size;
  unsigned int z = (id / rainbowmist_global_x_size) / rainbowmist_global_y_size;
  if (rainbowmist_global_z_size <= z) {
#if defined(RAINBOWMIST_CPP11) || defined(RAINBOWMIST_CUDA)
    printf("Global ID is overflow ([%d,%d,%d] over [%d,%d,%d)\n", x, y, z,
           rainbowmist_global_x_size, rainbowmist_global_y_size,
           rainbowmist_global_z_size);
#endif
    x = y = z = 0;
  }
  return uvec3(x, y, z);
}

inline vec2 make_vec2(float a, float b) {
  vec2 ret;
  ret.x = a;
  ret.y = b;
  return ret;
}

inline vec3 make_vec3(float a, float b, float c) {
  vec3 ret;
  ret.x = a;
  ret.y = b;
  ret.z = c;
  return ret;
}

inline vec4 make_vec4(float a, float b, float c, float d) {
  vec4 ret;
  ret.x = a;
  ret.y = b;
  ret.z = c;
  ret.w = d;
  return ret;
}

inline vec2 vmix(const vec2 x, const vec2 y, const float a)
{
  return x + (y - x) * a;
}


#endif

#endif  // RAINBOWMIST_H_
