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

#include <float.h> // import FLT_EPSILON

// From CUDA SDK samples
#include "helper_math.h"

#define DEVICE __device__
#define HOST   __host__
#define GLOBAL __global__
#define LOCAL  __local__
#define KERNEL __device__

#define vnormalize(x) normalize(x)
#define vcross(a, b) cross(a, b)
#define vdot(a, b) dot(a, b)


#elif defined(OPENCL)  // NOTE(LTE): Application must pass `-D OPENCL` as a OpenCL compile flags.

// We only support OpenCL 1.2

#define RAINBOWMIST_OPENCL (1)

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

static inline float2 make_float2(float a, float b)
{
  return (float2)(a, b);
}

static inline float3 make_float3(float a, float b, float c)
{
  return (float3)(a, b, c);
}

static inline float4 make_float4(float a, float b, float c, float d)
{
  return (float4)(a, b, c, d);
}

#else // C++

#define RAINBOWMIST_CPP11 (1)

#include <cfloat>
#include <cmath>

// TODO(LTE): Remove glm dependency
#define RAINBOWMIST_USE_GLM (1)

#ifdef RAINBOWMIST_USE_GLM

#include "glm/glm.hpp"
#include "glm/geometric.hpp"

typedef glm::vec2 float2;
typedef glm::vec3 float3;
typedef glm::vec4 float4;

#define vnormalize(x) normalize(x)
#define vcross(a, b) cross(a, b)
#define vdot(a, b) dot(a, b)

#else

// ----------------------------------------------------
template <typename T = float>
class real2 {
 public:
  real2() {}
  real2(T x) {
    v[0] = x;
    v[1] = x;
  }
  real2(T xx, T yy) {
    v[0] = xx;
    v[1] = yy;
  }
  explicit real2(const T *p) {
    v[0] = p[0];
    v[1] = p[1];
  }

  inline T x() const { return v[0]; }
  inline T y() const { return v[1]; }

  real2 operator*(T f) const { return real2(x() * f, y() * f); }
  real2 operator-(const real2 &f2) const {
    return real3(x() - f2.x(), y() - f2.y());
  }
  real2 operator*(const real2 &f2) const {
    return real3(x() * f2.x(), y() * f2.y());
  }
  real2 operator+(const real2 &f2) const {
    return real3(x() + f2.x(), y() + f2.y());
  }
  real2 &operator+=(const real2 &f2) {
    v[0] += f2.x();
    v[1] += f2.y();
    return (*this);
  }
  real2 operator/(const real2 &f2) const {
    return real3(x() / f2.x(), y() / f2.y());
  }
  real2 operator-() const {
    return real2(-x(), -y());
  }
  T operator[](int i) const { return v[i]; }
  T &operator[](int i) { return v[i]; }

  T v[2];
  // T pad[2];  // for alignment(when T = float)
};

template <typename T = float>
class real3 {
 public:
  real3() {}
  real3(T x) {
    v[0] = x;
    v[1] = x;
    v[2] = x;
  }
  real3(T xx, T yy, T zz) {
    v[0] = xx;
    v[1] = yy;
    v[2] = zz;
  }
  explicit real3(const T *p) {
    v[0] = p[0];
    v[1] = p[1];
    v[2] = p[2];
  }

  inline T x() const { return v[0]; }
  inline T y() const { return v[1]; }
  inline T z() const { return v[2]; }

  real3 operator*(T f) const { return real3(x() * f, y() * f, z() * f); }
  real3 operator-(const real3 &f2) const {
    return real3(x() - f2.x(), y() - f2.y(), z() - f2.z());
  }
  real3 operator*(const real3 &f2) const {
    return real3(x() * f2.x(), y() * f2.y(), z() * f2.z());
  }
  real3 operator+(const real3 &f2) const {
    return real3(x() + f2.x(), y() + f2.y(), z() + f2.z());
  }
  real3 &operator+=(const real3 &f2) {
    v[0] += f2.x();
    v[1] += f2.y();
    v[2] += f2.z();
    return (*this);
  }
  real3 operator/(const real3 &f2) const {
    return real3(x() / f2.x(), y() / f2.y(), z() / f2.z());
  }
  real3 operator-() const {
    return real3(-x(), -y(), -z());
  }
  T operator[](int i) const { return v[i]; }
  T &operator[](int i) { return v[i]; }

  T v[3];
  // T pad;  // for alignment(when T = float)
};

template <typename T>
inline real3<T> operator*(T f, const real3<T> &v) {
  return real3<T>(v.x() * f, v.y() * f, v.z() * f);
}

template <typename T>
inline real3<T> vneg(const real3<T> &rhs) {
  return real3<T>(-rhs.x(), -rhs.y(), -rhs.z());
}

template <typename T>
inline T vlength(const real3<T> &rhs) {
  return std::sqrt(rhs.x() * rhs.x() + rhs.y() * rhs.y() + rhs.z() * rhs.z());
}

template <typename T>
inline real3<T> vnormalize(const real3<T> &rhs) {
  real3<T> v = rhs;
  T len = vlength(rhs);
  if (fabs(len) > 1.0e-6f) {
    float inv_len = 1.0f / len;
    v.v[0] *= inv_len;
    v.v[1] *= inv_len;
    v.v[2] *= inv_len;
  }
  return v;
}

template <typename T>
inline real3<T> vcross(real3<T> a, real3<T> b) {
  real3<T> c;
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
  return c;
}

template <typename T>
inline T vdot(real3<T> a, real3<T> b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T = float>
class real4 {
 public:
  real4() {}
  real4(T x) {
    v[0] = x;
    v[1] = x;
    v[2] = x;
    v[3] = x;
  }
  real4(T xx, T yy, T zz, T ww) {
    v[0] = xx;
    v[1] = yy;
    v[2] = zz;
    v[3] = ww;
  }
  explicit real4(const T *p) {
    v[0] = p[0];
    v[1] = p[1];
    v[2] = p[2];
    v[3] = p[3];
  }

  inline T x() const { return v[0]; }
  inline T y() const { return v[1]; }
  inline T z() const { return v[2]; }
  inline T w() const { return v[3]; }

  real4 operator*(T f) const { return real4(x() * f, y() * f, z() * f, w() * f); }
  real4 operator-(const real4 &f2) const {
    return real4(x() - f2.x(), y() - f2.y(), z() - f2.z(), w() - f2.w());
  }
  real4 operator*(const real4 &f2) const {
    return real4(x() * f2.x(), y() * f2.y(), z() * f2.z(), w() * f2.w());
  }
  real4 operator+(const real4 &f2) const {
    return real4(x() + f2.x(), y() + f2.y(), z() + f2.z(), w() + f2.w());
  }
  real4 &operator+=(const real4 &f2) {
    v[0] += f2.x();
    v[1] += f2.y();
    v[2] += f2.z();
    v[3] += f2.w();
    return (*this);
  }
  real4 operator/(const real4 &f2) const {
    return real4(x() / f2.x(), y() / f2.y(), z() / f2.z(), w() / f2.w());
  }
  real4 operator-() const {
    return real4(-x(), -y(), -z(), -w());
  }
  T operator[](int i) const { return v[i]; }
  T &operator[](int i) { return v[i]; }

  T v[4];
};

typedef real2<float> float2;
typedef real3<float> float3;
typedef real4<float> float4;
typedef real2<double> double2;
typedef real3<double> double3;
typedef real4<double> double4;

// ----------------------------------------------------
#endif // RAINBOWMIST_USE_GLM

// NOTE(LTE): Liminatation
// - No shared variable support

#define KERNEL 
#define DEVICE 
#define GLOBAL 
#define SHARED ERROR


static inline float2 make_float2(float a, float b)
{
  return float2(a, b);
}

static inline float3 make_float3(float a, float b, float c)
{
  return float3(a, b, c);
}

static inline float4 make_float4(float a, float b, float c, float d)
{
  return float4(a, b, c, d);
}

#endif 

#endif // RAINBOWMIST_H_
