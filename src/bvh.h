#pragma once

#include "common.h"


// Triangle struct
typedef struct {
  float3 vertex0, vertex1, vertex2;
  float3 centroid;
} Tri;


// BVHNode struct and functions
typedef struct {
  float3 aabbMin, aabbMax;
  uint leftFirst, triCount;
} BVHNode;

int BVHNodeIsLeaf(BVHNode *node) {
  return node->triCount > 0;
}


// AABB struct and functions
typedef struct {
  float3 bmin;
  float3 bmax;
} AABB;

void AABBInit(AABB *aabb) {
  aabb->bmin = make_float3(1e30f);
  aabb->bmax = make_float3(-1e30f);
}

void AABBGrow (AABB *aabb, float3 *p) {
  // bmin = fminf(bmin, p)
  float3 *bmin = &aabb->bmin;
  bmin->x = fminf(bmin->x, p->x);
  bmin->y = fminf(bmin->y, p->y);
  bmin->z = fminf(bmin->z, p->z);

  // bmax = fmaxf(bmax, p)
  float3 *bmax = &aabb->bmax;
  bmax->x = fmaxf(bmax->x, p->x);
  bmax->y = fmaxf(bmax->y, p->y);
  bmax->z = fmaxf(bmax->z, p->z);
}

void AABBGrowAABB (AABB *aabb, AABB *b) {
  if (b->bmin.x != 1e30f) {
    // grow(b.bmin)
    AABBGrow(aabb, &b->bmin);
    // grow(b.bmax) 
    AABBGrow(aabb, &b->bmax);
  }
}

float AABBArea(AABB *aabb) {
  // float3 e = bmax - bmin;  // box extent
  float3 e = {
    aabb->bmax.x - aabb->bmin.x,
    aabb->bmax.y - aabb->bmin.y,
    aabb->bmax.z - aabb->bmin.z
  };
  return e.x * e.y + e.y * e.z + e.z * e.x;
}


// Bin struct
typedef struct {
  AABB bounds;
  int triCount;
} Bin;

void BinInit(Bin *bin) {
  AABBInit(&bin->bounds);
  bin->triCount = 0;
}


// Ray struct
typedef struct __attribute__((aligned(64))) {
  float3 O;
  float dummy1;

  float3 D;
  float dummy2;

  float3 rD;
  float dummy3;

  float t;
} Ray;

void RayInit(Ray *ray) {
  ray->O = make_float3(1.0f);
  ray->D = make_float3(1.0f);
  ray->rD = make_float3(1.0f);
  ray->t = 1e30f;
}