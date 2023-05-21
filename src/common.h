#pragma once

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

typedef unsigned int uint;
typedef long long ll;
typedef unsigned long long ull;

// macro
#define SCRWIDTH	1024
#define SCRHEIGHT	512

// global variables
ll flopcount = 0;

// float3 struct and functions
typedef struct {
  float x, y, z;
} float3;

float3 make_float3(float x) {
  float3 result;
  result.x = x;
  result.y = x;
  result.z = x;
  return result;
}

float3 make_float3_3(float x, float y, float z) {
  float3 result;
  result.x = x;
  result.y = y;
  result.z = z;
  return result;
}

// TODO(xiaoyuan) not very good function
float float3_at(float3 a, int i) {
  if (i == 0) return a.x;
  if (i == 1) return a.y;
  if (i == 2) return a.z;
  printf("float3_at error\n");
  return 0;
}


// math
float fminf(float a, float b) { return a < b ? a : b; }
float fmaxf(float a, float b) { return a > b ? a : b; }
float rsqrtf(float x) { return 1.0f / sqrtf(x); }
float sqrf(float x) { return x * x; }
int sqr(int x) { return x * x; }

float3 fminf_float3(const float3 *a, const float3 *b) { return make_float3_3(fminf(a->x, b->x), fminf(a->y, b->y), fminf(a->z, b->z)); }
float3 fmaxf_float3(const float3 *a, const float3 *b) { return make_float3_3(fmaxf(a->x, b->x), fmaxf(a->y, b->y), fmaxf(a->z, b->z)); }


// float3 operations
// TODO(xiaoyuan) do we need to use inline for these simple functions?
// TODO(xiaoyuan) do we need to use pointer even here
float3 SubFloat3(float3 a, float3 b) {
  float3 result;
  result.x = a.x - b.x;
  result.y = a.y - b.y;
  result.z = a.z - b.z;
  return result;
}

float3 AddFloat3(float3 a, float3 b) {
  float3 result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.z = a.z + b.z;
  return result;
}

float3 MulConstFloat3(float3 a, float b) {
  float3 result;
  result.x = a.x * b;
  result.y = a.y * b;
  result.z = a.z * b;
  return result;
}

float3 cross(const float3 a, const float3 b) {
  float3 result;
  result.x = a.y * b.z - a.z * b.y;
  result.y = a.z * b.x - a.x * b.z;
  result.z = a.x * b.y - a.y * b.x;
  return result;
}

float dot(const float3 a, const float3 b ) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 normalize(const float3 v) {
  float invLen = rsqrtf(dot(v, v));
  return MulConstFloat3(v, invLen);
}

// random
uint seed = 0x12345678;
uint RandomUInt(void) {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return seed;
}
uint RandomUIntSeed(uint seed) {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return seed;
}
float RandomFloat(void) { return RandomUInt() * 2.3283064365387e-10f; }
float RandomFloatSeed(uint seed) { return RandomUIntSeed(seed) * 2.3283064365387e-10f; }
float Rand(float range) { return RandomFloat() * range; }