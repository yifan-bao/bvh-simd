#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h> // AVX intrinsics


#include "bvh.h"
#include "utils.h"
#include "argparse.h"
#include "tsc_x86.h"

#define FREQUENCY 2.0e9  // disable turbo boost
// optimization used
// 1. SAH
// 2. blocking for imaging
// 3. binning
// TODO: SIMD, Data Layout, Numerical Issue
// TODO: Topology Optimization, Spatial Split

// bin count
#define BINS 8

// Config
typedef struct {
  int trinumber;
  int doValid;
  const char *path;
  const char *save_path;
} Config;

// TODO(xiaoyuan) init these variables
typedef struct {
  Tri *tri;  // now coexisting for easy implementation
  float* vertex0_x, *vertex0_y, *vertex0_z;
  float* vertex1_x, *vertex1_y, *vertex1_z;
  float* vertex2_x, *vertex2_y, *vertex2_z;
  float* centroid_x, *centroid_y, *centroid_z;
  uint *triIdx;
  BVHNode *bvhNode;
  uint rootNodeIdx;
  uint nodesUsed;
  ull N;
} BVHTree;

// forward declarations
void Subdivide(BVHTree *tree, uint nodeIdx);
void UpdateNodeBounds(BVHTree *tree, uint nodeIdx);

// functions
void IntersectTri(Ray *ray, uint triCount,  __m256 vector0_x, __m256 vector0_y, __m256 vector0_z, __m256 vector1_x, __m256 vector1_y, __m256 vector1_z, __m256 vector2_x, __m256 vector2_y, __m256 vector2_z) {
  // Create a mask to enable/disable elements beyond the desired count - inefficient
  __m256 mask = _mm256_setr_ps(
        triCount > 0 ? -1 : 0,
        triCount > 1 ? -1 : 0,
        triCount > 2 ? -1 : 0,
        triCount > 3 ? -1 : 0,
        triCount > 4 ? -1 : 0,
        triCount > 5 ? -1 : 0,
        triCount > 6 ? -1 : 0,
        triCount > 7 ? -1 : 0
    );
    // Use the mask to selectively keep the desired elements
    __m256 edge1_x =  _mm256_sub_ps(vector1_x, vector0_x);
    __m256 edge1_y =  _mm256_sub_ps(vector1_y, vector0_y);
    __m256 edge1_z =  _mm256_sub_ps(vector1_z, vector0_z);

    __m256 edge2_x =  _mm256_sub_ps(vector2_x, vector0_x);
    __m256 edge2_y =  _mm256_sub_ps(vector2_y, vector0_y);
    __m256 edge2_z =  _mm256_sub_ps(vector2_z, vector0_z);

    // cross;
    __m256 vector_D_x = _mm256_set1_ps(ray->D.x);
    __m256 vector_D_y = _mm256_set1_ps(ray->D.y);
    __m256 vector_D_z = _mm256_set1_ps(ray->D.z);
    __m256 h_x = _mm256_sub_ps(_mm256_mul_ps(vector_D_y, edge2_z), _mm256_mul_ps(vector_D_z, edge2_y));
    __m256 h_y = _mm256_sub_ps(_mm256_mul_ps(vector_D_z, edge2_x), _mm256_mul_ps(vector_D_x, edge2_z));
    __m256 h_z = _mm256_sub_ps(_mm256_mul_ps(vector_D_x, edge2_y), _mm256_mul_ps(vector_D_y, edge2_x));
    
    // dot
    __m256 a =  _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(edge1_x, h_x), _mm256_mul_ps(edge1_y, h_y)), _mm256_mul_ps(edge1_z, h_z));

    // constant packed single for comparison
    __m256 minus_00001f = _mm256_set1_ps(-0.0001f);
    __m256 plus_00001f = _mm256_set1_ps(0.0001f);
    __m256 zeros = _mm256_setzero_ps();
    __m256 ones = _mm256_set1_ps(1.0f);

    __m256 maskLowera = _mm256_cmp_ps(a, minus_00001f, _CMP_GT_OQ); // a > -0.0001f true
    __m256 maskUppera = _mm256_cmp_ps(a, plus_00001f, _CMP_LT_OQ);  // a < 0.0001f true
    __m256 resultMaska = _mm256_and_ps(maskLowera, maskUppera);     // all 1 means satisfy the if condition

    __m256 f = _mm256_div_ps(_mm256_set1_ps(1.0f), a);
    __m256 s_x = _mm256_sub_ps(_mm256_set1_ps(ray->O.x), vector0_x);
    __m256 s_y = _mm256_sub_ps(_mm256_set1_ps(ray->O.y), vector0_y);
    __m256 s_z = _mm256_sub_ps(_mm256_set1_ps(ray->O.z), vector0_z);

    // dot 
    __m256 u = _mm256_mul_ps(f, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(s_x, h_x), _mm256_mul_ps(s_y, h_y)), _mm256_mul_ps(s_z, h_z))); 

    __m256 maskLoweru = _mm256_cmp_ps(u, zeros, _CMP_LT_OQ); // u < 0.0f true
    __m256 maskUpperu = _mm256_cmp_ps(u, ones, _CMP_GT_OQ); // u > 1.0f true
    __m256 resultMasku = _mm256_or_ps(maskLoweru, maskUpperu);

    // cross
    __m256 q_x = _mm256_sub_ps(_mm256_mul_ps(s_y, edge1_z), _mm256_mul_ps(s_z, edge1_y));
    __m256 q_y = _mm256_sub_ps(_mm256_mul_ps(s_z, edge1_x), _mm256_mul_ps(s_x, edge1_z));
    __m256 q_z = _mm256_sub_ps(_mm256_mul_ps(s_x, edge1_y), _mm256_mul_ps(s_y, edge1_x));

    // dot
    __m256 v = _mm256_mul_ps(f, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(vector_D_x, q_x), _mm256_mul_ps(vector_D_y, q_y)), _mm256_mul_ps(vector_D_z, q_z))); 

    __m256 mask1 = _mm256_cmp_ps(v, zeros, _CMP_LT_OQ);
    __m256 mask2 = _mm256_cmp_ps(_mm256_add_ps(u, v), ones, _CMP_GT_OQ);
    __m256 resultMaskuv = _mm256_or_ps(mask1, mask2);
    __m256 t = _mm256_mul_ps(f, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(edge2_x, q_x), _mm256_mul_ps(edge2_y, q_y)), _mm256_mul_ps(edge2_z, q_z))); 

    // process t using all the masks. If mask return,t to infinity. If exceed the number, t to infinity. 
    __m256 setValue = _mm256_set1_ps(1e30f);
    // Use blending to conditionally set the value of t
    t = _mm256_blendv_ps(t, setValue, resultMaska);
    t = _mm256_blendv_ps(t, setValue, resultMasku);
    t = _mm256_blendv_ps(t, setValue, resultMaskuv);
    t = _mm256_blendv_ps(t, setValue, mask);

    __m256 resultMaskt = _mm256_cmp_ps(t, _mm256_set1_ps(0.0001f), _CMP_GT_OQ);
    t = _mm256_blendv_ps(setValue, t, resultMaskt);  // t > 0.0001f, true.

    __m256 temp_t = _mm256_set1_ps(ray->t); 
    temp_t = _mm256_min_ps(temp_t, t); // get the minimum

    // reduce to one min back. Try to avoid memory access
    // temp_t = _mm256_min_ps(temp_t, _mm256_permute_ps(temp_t, _MM_SHUFFLE(2, 3, 0, 1)));
    // temp_t = _mm256_min_ps(temp_t, _mm256_permute_ps(temp_t, _MM_SHUFFLE(1, 0, 3, 2)));
    // temp_t = _mm256_min_ps(temp_t, _mm256_permute2f128_ps(temp_t, temp_t, 0x01));
    // temp_t = _mm256_min_ps(temp_t, _mm256_permute_ps(temp_t, _MM_SHUFFLE(0, 1, 2, 3)));
    // temp_t = _mm256_min_ps(temp_t, _mm256_permute2f128_ps(temp_t, temp_t, 0x01));
    // ray->t = _mm256_cvtss_f32(temp_t); // store the minimum to ray->t

    // another reduce method
    __m128 low = _mm256_castps256_ps128(temp_t);
    __m128 high = _mm256_extractf128_ps(temp_t, 1);
    __m128 min_4 = _mm_max_ps(low, high);
    __m128 min_2 = _mm_min_ps(min_4, _mm_shuffle_ps(min_4, min_4, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128 min_1 = _mm_min_ss(min_2, _mm_shuffle_ps(min_2, min_2, _MM_SHUFFLE(0, 3, 2, 1)));
    _mm_store_ss(&(ray->t), min_1);

}

// Intersect with 2 aabb boxes (adjacent child)
void IntersectAABB_AVX(const Ray *ray, __m256 bmin8, __m256 bmax8, float* dist1, float* dist2) {
    // static const mask8 = manually calculated mask is better
    const __m256 mask8 = _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_set_ps(1, 0, 0, 0, 1, 0, 0, 0), _CMP_EQ_OQ);
    
    __m256 source = _mm256_castps128_ps256(_mm_loadu_ps(&(ray->O.x))); // _mm_loadu_ps: _mm128
    // Duplicate the __m256 variable into two copies
    __m256 ray_O8 = _mm256_permute2f128_ps(source, source, 0x00);
    source = _mm256_castps128_ps256(_mm_loadu_ps(&(ray->rD.x)));
    __m256 ray_rD8 = _mm256_permute2f128_ps(source, source, 0x00);

    __m256 t1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_and_ps(bmin8, mask8), ray_O8), ray_rD8);
    __m256 t2 = _mm256_mul_ps(_mm256_sub_ps(_mm256_and_ps(bmax8, mask8), ray_O8), ray_rD8);
    __m256 vmax8 = _mm256_max_ps(t1, t2), vmin8 = _mm256_min_ps(t1, t2);

    
    // TODO(yifan): change the lazy implementation. Try to not use __m128
    __m128 vmax4 = _mm256_extractf128_ps(vmax8, 0);
    __m128 vmin4 = _mm256_extractf128_ps(vmin8, 0);
    float tmax = fmin(fmin(_mm_cvtss_f32(vmax4), fmin(_mm_cvtss_f32(_mm_shuffle_ps(vmax4, vmax4, 0x55)), _mm_cvtss_f32(_mm_shuffle_ps(vmax4, vmax4, 0xAA)))),
                          fmin(_mm_cvtss_f32(_mm_shuffle_ps(vmax4, vmax4, 0xFF)), _mm_cvtss_f32(_mm_shuffle_ps(vmax4, vmax4, 0x00))));

    float tmin = fmax(fmax(_mm_cvtss_f32(vmin4), fmax(_mm_cvtss_f32(_mm_shuffle_ps(vmin4, vmin4, 0x55)), _mm_cvtss_f32(_mm_shuffle_ps(vmin4, vmin4, 0xAA)))),
                          fmax(_mm_cvtss_f32(_mm_shuffle_ps(vmin4, vmin4, 0xFF)), _mm_cvtss_f32(_mm_shuffle_ps(vmin4, vmin4, 0x00))));
    
    vmax4 = _mm256_extractf128_ps(vmax8, 1);
    vmin4 = _mm256_extractf128_ps(vmin8, 1);
    float tmax_1 = fmin(fmin(_mm_cvtss_f32(vmax4), fmin(_mm_cvtss_f32(_mm_shuffle_ps(vmax4, vmax4, 0x55)), _mm_cvtss_f32(_mm_shuffle_ps(vmax4, vmax4, 0xAA)))),
                          fmin(_mm_cvtss_f32(_mm_shuffle_ps(vmax4, vmax4, 0xFF)), _mm_cvtss_f32(_mm_shuffle_ps(vmax4, vmax4, 0x00))));

    float tmin_1 = fmax(fmax(_mm_cvtss_f32(vmin4), fmax(_mm_cvtss_f32(_mm_shuffle_ps(vmin4, vmin4, 0x55)), _mm_cvtss_f32(_mm_shuffle_ps(vmin4, vmin4, 0xAA)))),
                          fmax(_mm_cvtss_f32(_mm_shuffle_ps(vmin4, vmin4, 0xFF)), _mm_cvtss_f32(_mm_shuffle_ps(vmin4, vmin4, 0x00))));
    
    if (tmax >= tmin && tmin < ray->t && tmax > 0) {
        *dist1 = tmin;
    }
    // else {
        
    // }
    if (tmax_1 >= tmin_1 && tmin_1 < ray->t && tmax_1 > 0) {
        *dist2 = tmin_1;
    }
}

void IntersectBVH(BVHTree *tree, Ray *ray) {
  BVHNode *node = tree->bvhNode + tree->rootNodeIdx;
  BVHNode *stack[64];
  uint stackPtr = 0;
  while (1) {
    if (BVHNodeIsLeaf(node)) {
      for (uint i = 0; i < node->triCount; i++) {
        // inefficient gathering of data. another way is to swap the data when spliting instead of swaping the index
        // try and compare the two.
        __m256 vector0_x, vector0_y, vector0_z, vector1_x, vector1_y, vector1_z, vector2_x, vector2_y, vector2_z;
        __m256i indexVector;
        indexVector = _mm256_loadu_si256((__m256i*)(tree->triIdx + node->leftFirst)); // load index vector 
        vector0_x = _mm256_i32gather_ps(tree->vertex0_x, indexVector, sizeof(float)); // gather data using index vector
        vector0_y = _mm256_i32gather_ps(tree->vertex0_y, indexVector, sizeof(float));
        vector0_z = _mm256_i32gather_ps(tree->vertex0_z, indexVector, sizeof(float));
        vector1_x = _mm256_i32gather_ps(tree->vertex1_x, indexVector, sizeof(float));
        vector1_y = _mm256_i32gather_ps(tree->vertex1_y, indexVector, sizeof(float));
        vector1_z = _mm256_i32gather_ps(tree->vertex1_z, indexVector, sizeof(float));
        vector2_x = _mm256_i32gather_ps(tree->vertex2_x, indexVector, sizeof(float));
        vector2_y = _mm256_i32gather_ps(tree->vertex2_y, indexVector, sizeof(float));
        vector2_z = _mm256_i32gather_ps(tree->vertex2_z, indexVector, sizeof(float));
        
        // we need to consider the number is less than 8 or not. node->triCount used for masking
        IntersectTri(ray, node->triCount, vector0_x, vector0_y, vector0_z, vector1_x, vector1_y, vector1_z, vector2_x, vector2_y, vector2_z);
      }
      if (stackPtr == 0) {
        break;
      }
      else {
        node = stack[--stackPtr];
      }
      continue;
    }
    BVHNode* child1 = tree->bvhNode + node->leftFirst;
    BVHNode* child2 = tree->bvhNode + (node->leftFirst + 1);
    // aabbMinMax
    __m256 bmin8 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_extractf128_ps(child1->aabbMinMax, 0)), _mm256_extractf128_ps(child2->aabbMinMax, 0) , 1);
    __m256 bmax8 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_extractf128_ps(child1->aabbMinMax, 1)), _mm256_extractf128_ps(child2->aabbMinMax, 1) , 1);
    float dist1 = 1e30f, dist2 = 1e30f;
    IntersectAABB_AVX(ray, bmin8, bmax8, &dist1, &dist2);
    // dist1 = IntersectAABB(ray, child1->aabbMin, child1->aabbMax);
    // dist2 = IntersectAABB(ray, child2->aabbMin, child2->aabbMax);
    
    if (dist1 > dist2) {
      swap_float(&dist1, &dist2);
      swap_bvhnode(child1, child2);
    }
    if (dist1 == 1e30f) {
      if (stackPtr == 0)
        break;
      else
        node = stack[--stackPtr];
    } else {
      node = child1;
      if (dist2 != 1e30f) stack[stackPtr++] = child2;
    }
  }
}

void BuildBVH(BVHTree *tree) {
#ifdef COUNTFLOPS
  printf("before_build_flops %llu\n", flopcount);
#endif

  // time
  myInt64 build_start, build_end;
  build_start = start_tsc();

  // create the BVH node pool
  tree->bvhNode = (BVHNode*)aligned_alloc(64, sizeof(BVHNode) * tree->N * 2);

  // populate triangle index array SIMD
  __m256i ids = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0); // assume ull is 64 bit
  uint index;
  for(index = 0; index < tree->N-7; index += 8) {
    _mm256_storeu_si256((__m256i*)(tree->triIdx + index), ids);
    _mm256_add_epi32(ids, _mm256_set1_epi32(8));
  }
  // cleanup code
  for(; index < tree->N; index++) {
    tree->triIdx[index] = index;
  }
  
  // calculate triangle centroids for partitioning
  __m256 one_third = _mm256_set1_ps(1.0f / 3.0f);
  ull i;
  for(i  = 0; i < tree->N - 7; i += 8) {
      __m256 v0x = _mm256_load_ps(tree->vertex0_x + i);
      __m256 v0y = _mm256_load_ps(tree->vertex0_y + i);
      __m256 v0z = _mm256_load_ps(tree->vertex0_z + i);
      __m256 v1x = _mm256_load_ps(tree->vertex1_x + i);
      __m256 v1y = _mm256_load_ps(tree->vertex1_y + i);
      __m256 v1z = _mm256_load_ps(tree->vertex1_z + i);
      __m256 v2x = _mm256_load_ps(tree->vertex2_x + i);
      __m256 v2y = _mm256_load_ps(tree->vertex2_y + i);
      __m256 v2z = _mm256_load_ps(tree->vertex2_z + i);

      __m256 centroid_x = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(v0x, v1x), v2x), one_third);
      __m256 centroid_y = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(v0y, v1y), v2y), one_third);
      __m256 centroid_z = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(v0z, v1z), v2z), one_third);
      _mm256_store_ps(tree->centroid_x + i, centroid_x);
      _mm256_store_ps(tree->centroid_y + i, centroid_y);
      _mm256_store_ps(tree->centroid_z + i, centroid_z);
  }
  // redundant
  for (ull j = 0; j < i; ++j) {
    Tri *tri = tree->tri + i;
    tri->centroid.x = tree->centroid_x[i];
    tri->centroid.y = tree->centroid_y[i];
    tri->centroid.z = tree->centroid_z[i];
  }
  // cleanup
  for(; i < tree->N; i += 8) {
    Tri *tri = tree->tri + i;
    tri->centroid = AddFloat3(AddFloat3(tri->vertex0, tri->vertex1), tri->vertex2);
    tri->centroid = MulConstFloat3(tri->centroid, 1.0f / 3.0f);
    tree->centroid_x[i] = tri->centroid.x;
    tree->centroid_y[i] = tri->centroid.y;
    tree->centroid_z[i] = tri->centroid.z;
  }
  
  // assign all triangles to root node
  BVHNode *root = tree->bvhNode + tree->rootNodeIdx;
  root->leftFirst = 0;
  root->triCount = tree->N;

  UpdateNodeBounds(tree, tree->rootNodeIdx);
  // subdivide recursively
  Subdivide(tree, tree->rootNodeIdx);
  
  build_end = stop_tsc(build_start);
  // TODO(xiaoyuan): think of overflow problem
  printf("build_cycles %llu\n", build_end);
#ifdef COUNTFLOPS
  printf("build_flops %llu\n", flopcount);
#endif
  printf("build_nodes %d\n", tree->nodesUsed);
}

void UpdateNodeBounds(BVHTree *tree, uint nodeIdx) {
  // old version
  BVHNode *node = tree->bvhNode + nodeIdx;
  node->aabbMin = make_float3(1e30f);
  node->aabbMax = make_float3(-1e30f);
  for (uint first = node->leftFirst, i = 0; i < node->triCount; i++) {
    uint leafTriIdx = tree->triIdx[first + i];
    Tri *leafTri = tree->tri + leafTriIdx;
    // TODO: inline or use question operator
    node->aabbMin = fminf_float3(&node->aabbMin, &leafTri->vertex0);  
    node->aabbMin = fminf_float3(&node->aabbMin, &leafTri->vertex1);
    node->aabbMin = fminf_float3(&node->aabbMin, &leafTri->vertex2);
    node->aabbMax = fmaxf_float3(&node->aabbMax, &leafTri->vertex0);
    node->aabbMax = fmaxf_float3(&node->aabbMax, &leafTri->vertex1);
    node->aabbMax = fmaxf_float3(&node->aabbMax, &leafTri->vertex2);
  }


  // unknown segmentation fault
  // BVHNode *node = tree->bvhNode + nodeIdx;
  // node->aabbMin = make_float3(1e30f);
  // node->aabbMax = make_float3(-1e30f);

  // __m256 vector0_x, vector0_y, vector0_z, vector1_x, vector1_y, vector1_z, vector2_x, vector2_y, vector2_z;
  // __m256i indexVector;
  // __m256 min_temp_x = _mm256_set1_ps(1e30f), min_temp_y = _mm256_set1_ps(1e30f), min_temp_z = _mm256_set1_ps(1e30f); 
  // __m256 max_temp_x = _mm256_set1_ps(-1e30f), max_temp_y = _mm256_set1_ps(-1e30f), max_temp_z = _mm256_set1_ps(-1e30f);
  
  // uint i=0;
  // uint first = node->leftFirst;
  // for (i = 0; i < node->triCount - 7; i += 8) {
  //   // inefficient gathering of data. If we swap the triangles when build then the data is continuous
  //   // but swap time is longer
  //   indexVector = _mm256_loadu_si256((__m256i*)(tree->triIdx + first + i)); // load index vector
  //   vector0_x = _mm256_i32gather_ps(tree->vertex0_x, indexVector, sizeof(float)); // gather data using index vector
  //   vector0_y = _mm256_i32gather_ps(tree->vertex0_y, indexVector, sizeof(float));
  //   vector0_z = _mm256_i32gather_ps(tree->vertex0_z, indexVector, sizeof(float));
  //   vector1_x = _mm256_i32gather_ps(tree->vertex1_x, indexVector, sizeof(float));
  //   vector1_y = _mm256_i32gather_ps(tree->vertex1_y, indexVector, sizeof(float));
  //   vector1_z = _mm256_i32gather_ps(tree->vertex1_z, indexVector, sizeof(float));
  //   vector2_x = _mm256_i32gather_ps(tree->vertex2_x, indexVector, sizeof(float));
  //   vector2_y = _mm256_i32gather_ps(tree->vertex2_y, indexVector, sizeof(float));
  //   vector2_z = _mm256_i32gather_ps(tree->vertex2_z, indexVector, sizeof(float));

  //   min_temp_x = _mm256_min_ps(_mm256_min_ps(_mm256_min_ps(vector0_x, vector1_x), vector2_x), min_temp_x);
  //   min_temp_y = _mm256_min_ps(_mm256_min_ps(_mm256_min_ps(vector0_y, vector1_y), vector2_y), min_temp_y);
  //   min_temp_z = _mm256_min_ps(_mm256_min_ps(_mm256_min_ps(vector0_z, vector1_z), vector2_z), min_temp_z);

  //   max_temp_x = _mm256_max_ps(_mm256_max_ps(_mm256_max_ps(vector0_x, vector1_x), vector2_x),max_temp_x);
  //   max_temp_y = _mm256_max_ps(_mm256_max_ps(_mm256_max_ps(vector0_y, vector1_y), vector2_y),max_temp_y);
  //   max_temp_z = _mm256_max_ps(_mm256_max_ps(_mm256_max_ps(vector0_z, vector1_z), vector2_z),max_temp_z);
  // }
  
  // // // cleanup code
  // float3 aabb_min = make_float3(1e30f);
  // float3 aabb_max = make_float3(-1e30f);
  // for (; i < node->triCount; i++) {
  //   uint leafTriIdx = tree->triIdx[first + i];
  //   Tri *leafTri = tree->tri + leafTriIdx;
  //   aabb_min = fminf_float3(&aabb_min, &leafTri->vertex0);
  //   aabb_min = fminf_float3(&aabb_min, &leafTri->vertex1);
  //   aabb_min = fminf_float3(&aabb_min, &leafTri->vertex2);
  //   aabb_max = fmaxf_float3(&aabb_max, &leafTri->vertex0);
  //   aabb_max = fmaxf_float3(&aabb_max, &leafTri->vertex1);
  //   aabb_max = fmaxf_float3(&aabb_max, &leafTri->vertex2);
  // }
  
  // if(node->triCount >= 8) {
  // // // reduce to one minimum
  // // // Compare lower and upper 4 floats and store minimums
  //   __m128 low = _mm256_castps256_ps128(min_temp_x);
  //   __m128 high = _mm256_extractf128_ps(min_temp_x, 1);
  //   __m128 min_4 = _mm_min_ps(low, high);
  //   // Shuffle and find minimum
  //   __m128 min_2 = _mm_min_ps(min_4, _mm_shuffle_ps(min_4, min_4, _MM_SHUFFLE(1, 0, 3, 2)));
  //   __m128 min_1 = _mm_min_ss(min_2, _mm_shuffle_ps(min_2, min_2, _MM_SHUFFLE(0, 3, 2, 1)));
  //   _mm_store_ss(&(node->aabbMin.x), min_1);
   
  //   low = _mm256_castps256_ps128(min_temp_y);
  //   high = _mm256_extractf128_ps(min_temp_y, 1);
  //   min_4 = _mm_min_ps(low, high);
  //   min_2 = _mm_min_ps(min_4, _mm_shuffle_ps(min_4, min_4, _MM_SHUFFLE(1, 0, 3, 2)));
  //   min_1 = _mm_min_ss(min_2, _mm_shuffle_ps(min_2, min_2, _MM_SHUFFLE(0, 3, 2, 1)));
  //   _mm_store_ss(&(node->aabbMin.y), min_1);

  //   low = _mm256_castps256_ps128(min_temp_z);
  //   high = _mm256_extractf128_ps(min_temp_z, 1);
  //   min_4 = _mm_min_ps(low, high);
  //   min_2 = _mm_min_ps(min_4, _mm_shuffle_ps(min_4, min_4, _MM_SHUFFLE(1, 0, 3, 2)));
  //   min_1 = _mm_min_ss(min_2, _mm_shuffle_ps(min_2, min_2, _MM_SHUFFLE(0, 3, 2, 1)));
  //   _mm_store_ss(&(node->aabbMin.z), min_1);
    
  //   // reduce to one maximum
  //   // Compare lower and upper 4 floats and store minimums
  //   low = _mm256_castps256_ps128(max_temp_x);
  //   high = _mm256_extractf128_ps(max_temp_x, 1);
  //   __m128 max_4 = _mm_max_ps(low, high);
  //   // Shuffle and find minimum
  //   __m128 max_2 = _mm_min_ps(max_4, _mm_shuffle_ps(max_4, max_4, _MM_SHUFFLE(1, 0, 3, 2)));
  //   __m128 max_1 = _mm_min_ss(max_2, _mm_shuffle_ps(max_2, max_2, _MM_SHUFFLE(0, 3, 2, 1)));
  //   // Extract minimum value as a float
  //   _mm_store_ss(&(node->aabbMax.x), max_1);

  //   low = _mm256_castps256_ps128(max_temp_y);
  //   high = _mm256_extractf128_ps(max_temp_y, 1);
  //   max_4 = _mm_max_ps(low, high);
  //   max_2 = _mm_min_ps(max_4, _mm_shuffle_ps(max_4, max_4, _MM_SHUFFLE(1, 0, 3, 2)));
  //   max_1 = _mm_min_ss(max_2, _mm_shuffle_ps(max_2, max_2, _MM_SHUFFLE(0, 3, 2, 1)));
  //   _mm_store_ss(&(node->aabbMax.y), max_1);

  //   low = _mm256_castps256_ps128(max_temp_z);
  //   high = _mm256_extractf128_ps(max_temp_z, 1);
  //   max_4 = _mm_max_ps(low, high);
  //   max_2 = _mm_min_ps(max_4, _mm_shuffle_ps(max_4, max_4, _MM_SHUFFLE(1, 0, 3, 2)));
  //   max_1 = _mm_min_ss(max_2, _mm_shuffle_ps(max_2, max_2, _MM_SHUFFLE(0, 3, 2, 1)));
  //   _mm_store_ss(&(node->aabbMax.z), max_1);
  // }
  
  // node->aabbMin = fminf_float3(&aabb_min, &(node->aabbMin));
  // node->aabbMax = fmaxf_float3(&aabb_max, &(node->aabbMax));
}

float FindBestSplitPlane(BVHTree *tree, BVHNode *node, int *axis, float *splitPos) {
  float bestCost = 1e30f;
  for (int a = 0; a < 3; a++) {
    float boundsMin = 1e30f, boundsMax = -1e30f;
    for (uint i = 0; i < node->triCount; i++) {
      Tri *triangle = tree->tri + tree->triIdx[node->leftFirst + i];
      boundsMin = fmin(boundsMin, float3_at(triangle->centroid, a));
      boundsMax = fmax(boundsMax, float3_at(triangle->centroid, a));
    }
    if (boundsMin == boundsMax) continue;
    // populate the bins
    Bin bin[BINS];
    for (int i = 0; i < BINS; i++) {
      BinInit(&bin[i]);
    }
    float scale = BINS / (boundsMax - boundsMin);
    for (uint i = 0; i < node->triCount; i++) {
      Tri *triangle = tree->tri + tree->triIdx[node->leftFirst + i];
      int binIdx = min_int(BINS - 1, (int)((float3_at(triangle->centroid, a) - boundsMin) * scale));  // TODO: do we need the min function?
      bin[binIdx].triCount++;
      AABBGrow(&bin[binIdx].bounds, &triangle->vertex0);
      AABBGrow(&bin[binIdx].bounds, &triangle->vertex1);
      AABBGrow(&bin[binIdx].bounds, &triangle->vertex2);
    }
    // gather data for the 7 planes between the 8 bins
    float leftArea[BINS - 1], rightArea[BINS - 1];
    int leftCount[BINS - 1], rightCount[BINS - 1];
    AABB leftBox, rightBox;
    AABBInit(&leftBox);
    AABBInit(&rightBox);
    int leftSum = 0, rightSum = 0;
    // TODO: possibly have some problem.
    for (int i = 0; i < BINS - 1; i++) {
      leftSum += bin[i].triCount;
      leftCount[i] = leftSum;
      AABBGrowAABB(&leftBox, &bin[i].bounds);
      leftArea[i] = AABBArea(&leftBox);
      rightSum += bin[BINS - 1 - i].triCount;
      rightCount[BINS - 2 - i] = rightSum;
      AABBGrowAABB(&rightBox, &bin[BINS - 1 - i].bounds);
      rightArea[BINS - 2 - i] = AABBArea(&rightBox);
    }
    // calculate SAH cost for the 7 planes
    scale = (boundsMax - boundsMin) / BINS;
    for (int i = 0; i < BINS - 1; i++) {
      float planeCost = leftCount[i] * leftArea[i] + rightCount[i] * rightArea[i];
      if (planeCost < bestCost) {
        *axis = a;
        *splitPos = boundsMin + scale * (i + 1);
        bestCost = planeCost;
      }
    }
  }
  return bestCost;
}

// simple simd
float CalculateNodeCost(BVHNode *node) {
  __m128 upper_half = _mm256_extractf128_ps(node->aabbMinMax, 1);
  __m128 lower_half = _mm256_castps256_ps128(node->aabbMinMax);
  __m128 result = _mm_sub_ps(upper_half, lower_half);
  float e[4];
  _mm_store_ps(e, result);
  float surfaceArea = e[0] * e[1] + e[1] * e[2] + e[2] * e[0];  // e.x * e.y + e.y * e.z + e.z * e.x;
  return node->triCount * surfaceArea;
}

// subdivide, less than 8 leaves version - gpt
void Subdivide(BVHTree *tree, uint nodeIdx) {

  BVHNode *node = tree->bvhNode + nodeIdx;

  // Determine split axis using SAH
  int axis;
  float splitPos;
  float splitCost = FindBestSplitPlane(tree, node, &axis, &splitPos);

  // Calculate the cost of not splitting the node
  float nosplitCost = CalculateNodeCost(node);

  // Terminate recursion if the split cost is higher than the nosplit cost
  if (splitCost >= nosplitCost) return;

  // In-place partition
  uint i = node->leftFirst;
  uint j = i + node->triCount - 1;
  while (i <= j) {
    if (float3_at(tree->tri[tree->triIdx[i]].centroid, axis) < splitPos)
      i++;
    else
      swap_uint(tree->triIdx + i, tree->triIdx + j--);
  }

  // Abort split if one of the sides is empty
  uint leftCount = i - node->leftFirst;
  if (leftCount == 0 || leftCount == node->triCount) return;

  // Create child nodes
  if (node->triCount <= 8) {
    // Leaf node with 8 or fewer primitives
    node->leftFirst = node->leftFirst; // No change in leftFirst index
    node->triCount = leftCount; // Update triCount to reflect the number of primitives in the left partition
  } else {
    // Internal node
    uint leftChildIdx = tree->nodesUsed++;
    uint rightChildIdx = tree->nodesUsed++;

    tree->bvhNode[leftChildIdx].leftFirst = node->leftFirst;
    tree->bvhNode[leftChildIdx].triCount = leftCount;

    tree->bvhNode[rightChildIdx].leftFirst = i;
    tree->bvhNode[rightChildIdx].triCount = node->triCount - leftCount;

    node->leftFirst = leftChildIdx;
    node->triCount = 0;

    UpdateNodeBounds(tree, leftChildIdx);
    UpdateNodeBounds(tree, rightChildIdx);

    // Recurse
    Subdivide(tree, leftChildIdx);
    Subdivide(tree, rightChildIdx);
  }
}

void InitRandom(BVHTree* tree, int triCount) {
  // malloc
  tree->N = triCount;
  tree->tri = malloc(tree->N * sizeof(Tri));
  tree->triIdx = malloc(tree->N * sizeof(uint));
  // intialize a scene with N random triangles
  for (ull i = 0; i < tree->N; i++) {
    float3 r0 = make_float3_3(RandomFloat(), RandomFloat(), RandomFloat());
    float3 r1 = make_float3_3(RandomFloat(), RandomFloat(), RandomFloat());
    float3 r2 = make_float3_3(RandomFloat(), RandomFloat(), RandomFloat());
    tree->tri[i].vertex0 = SubFloat3(MulConstFloat3(r0, 9), make_float3(5));
    tree->tri[i].vertex1 = AddFloat3(tree->tri[i].vertex0, r1);
    tree->tri[i].vertex2 = AddFloat3(tree->tri[i].vertex0, r2);
  }
  BuildBVH(tree);
}


void Init(BVHTree *tree, const char* filename) {
  // TOOD(xiaoyuan): careful for overflow
  int triCount = 0;  // Initialize triangle count

  // Count the number of triangles in the file
  FILE* countFile = fopen(filename, "r");
  if (countFile == NULL) {
    printf("[ERROR] Failed to open file: %s\n", filename);
    exit(1);
  }

  float dummy;
  while (fscanf(countFile, "%f %f %f %f %f %f %f %f %f\n", &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy) == 9) {
    triCount++;
  }
  fclose(countFile);

  printf("num_tri %d\n", triCount);

  // Allocate memory for triangles and triangle indices
  tree->N = triCount;
  tree->tri = malloc(tree->N * sizeof(Tri));
  tree->triIdx = malloc(tree->N * sizeof(uint));

  tree->vertex0_x = malloc(tree->N * sizeof(float));
  tree->vertex0_y = malloc(tree->N * sizeof(float));
    tree->vertex0_z = malloc(tree->N * sizeof(float));
    tree->vertex1_x = malloc(tree->N * sizeof(float));
    tree->vertex1_y = malloc(tree->N * sizeof(float));
    tree->vertex1_z = malloc(tree->N * sizeof(float));
    tree->vertex2_x = malloc(tree->N * sizeof(float));
    tree->vertex2_y = malloc(tree->N * sizeof(float));
    tree->vertex2_z = malloc(tree->N * sizeof(float));
    tree->centroid_x = malloc(tree->N * sizeof(float));
    tree->centroid_y = malloc(tree->N * sizeof(float));
    tree->centroid_z = malloc(tree->N * sizeof(float));


  // Read triangle data from file
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    printf("[error] Failed to open file: %s\n", filename);
    exit(1);
  }

  for (int t = 0; t < triCount; t++) {
    if (fscanf(file, "%f %f %f %f %f %f %f %f %f\n", &tree->tri[t].vertex0.x, &tree->tri[t].vertex0.y, &tree->tri[t].vertex0.z,
               &tree->tri[t].vertex1.x, &tree->tri[t].vertex1.y, &tree->tri[t].vertex1.z, &tree->tri[t].vertex2.x, &tree->tri[t].vertex2.y,
               &tree->tri[t].vertex2.z) != 9) {
      printf("File format is incorrect\n");
      exit(1);
    }
    tree->vertex0_x[t] =  tree->tri[t].vertex0.x;
    tree->vertex0_y[t] =  tree->tri[t].vertex0.y;
    tree->vertex0_z[t] =  tree->tri[t].vertex0.z;
    tree->vertex1_x[t] =  tree->tri[t].vertex1.x;
    tree->vertex1_y[t] =  tree->tri[t].vertex1.y;
    tree->vertex1_z[t] =  tree->tri[t].vertex1.z;
    tree->vertex2_x[t] =  tree->tri[t].vertex2.x;
    tree->vertex2_y[t] =  tree->tri[t].vertex2.y;
    tree->vertex2_z[t] =  tree->tri[t].vertex2.z;

  }
  fclose(file);

  // Construct the BVH
  BuildBVH(tree);
}


void Traverse(BVHTree *tree, Config *config) {
  // save the 3d visual image to a file
	FILE* writefile;
  if (config->save_path != NULL) {
    writefile = fopen(config->save_path, "w");
  }

  float3 p0 = make_float3_3(100, -100, -40);
  float3 p1 = make_float3_3(100, 100, -40);
  float3 p2 = make_float3_3(-100, -100, -40);

  Ray ray;
  RayInit(&ray);

  // time
#ifdef COUNTFLOPS
  printf("before_traverse_flops %llu\n", flopcount);
#endif
  myInt64 traverse_start, traverse_end;
  traverse_start = start_tsc();

  for (int y = 0; y < SCRHEIGHT; y += 4) {
    for (int x = 0; x < SCRWIDTH; x += 4) {
      // tiling/blocking
      for (int v = 0; v < 4; v++)
        for (int u = 0; u < 4; u++) {
          ray.O = make_float3_3(0.0f, 0.0f, 200.0f);
          // float3 pixelPos = ray.O + p0 + (p1 - p0) * ((x + u) / (float)SCRWIDTH) + (p2 - p0) * ((y + v) / (float)SCRHEIGHT);
          float3 pixelPos = AddFloat3(ray.O, p0);
          pixelPos = AddFloat3(pixelPos, MulConstFloat3(SubFloat3(p1, p0), (x + u) / (float)SCRWIDTH));
          pixelPos = AddFloat3(pixelPos, MulConstFloat3(SubFloat3(p2, p0), (y + v) / (float)SCRHEIGHT));

          ray.D = normalize(SubFloat3(pixelPos, ray.O));
          ray.t = 1e30f;
          ray.rD = make_float3_3(1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z);
          IntersectBVH(tree, &ray);
          // TODO(xiaoyuan!): add a debug mode, to avoid such code affecting performance
          if (config->save_path == NULL)
            continue;
          if (ray.t < 1e30f) {
            float3 point = comp_targetpoint(ray.O, ray.D, ray.t);
            fprintf(writefile, "%d %d %f %f %f\n", x + u, y + v, point.x, point.y, point.z);
          } else {
            fprintf(writefile, "%d %d\n", x + u, y + v);
          }
        }
    }
  }

  traverse_end = stop_tsc(traverse_start);
  printf("traverse_cycles %llu\n", traverse_end);
#ifdef COUNTFLOPS
  printf("total_flops %llu\n", flopcount);
#endif

  if (config->save_path != NULL) {
    fclose(writefile);
  }
}

void InitOptions(Config *config) {
  config->trinumber = 0;
  config->doValid = 0;
  config->path = NULL;
  config->save_path = NULL;
}

int parse(Config *config, int argc, const char **argv) {
  static const char *const usages[] = {
#ifdef COUNTFLOPS
    "quick_count [options]",
#else
    "quick [options]",
#endif
    NULL,
  };

  struct argparse_option options[] = {
      OPT_HELP(),
      OPT_INTEGER('t', "trinumber", &config->trinumber, "random trinumber", NULL, 0, 0),
      OPT_BOOLEAN('v', "valid", &config->doValid, "validate the result", NULL, 0, 0),
      OPT_STRING('f', "file", &config->path, "read from tri file", NULL, 0, 0),
      OPT_STRING('s', "save", &config->save_path, "save result to file", NULL, 0, 0),
      OPT_END(),
  };

  struct argparse argparse;
  argparse_init(&argparse, options, usages, 0);
  argparse_describe(&argparse, "ASL Team09 Project: BVH.", NULL);
  argc = argparse_parse(&argparse, argc, argv);
  if (config->trinumber != 0) printf("[Config] trinumber: %d\n", config->trinumber);
  if (config->doValid != 0) printf("[Config] doValid: %d\n", config->doValid);
  if (config->path != NULL) printf("[Config] path: %s\n", config->path);
  if (config->save_path != NULL) printf("[Config] save_path: %s\n", config->save_path);
  return 0;
}

int main(int argc, const char* argv[]) {
  const char *bin_name = argv[0];

  BVHTree tree = {
    .N = 0,
    .bvhNode = NULL,
    .tri = NULL,
    .vertex0_x = NULL,
    .vertex0_y = NULL,
    .vertex0_z = NULL,
    .vertex1_x = NULL,
    .vertex1_y = NULL,
    .vertex1_z = NULL,
    .vertex2_x = NULL,
    .vertex2_y = NULL,
    .vertex2_z = NULL,
    .centroid_x = NULL,
    .centroid_y = NULL, 
    .centroid_z = NULL,
    .triIdx = NULL,
    .rootNodeIdx = 0,
    .nodesUsed = 2,
  };

  // parse options
  Config config;
  InitOptions(&config);
  parse(&config, argc, argv);

  if (config.path != NULL) {
    Init(&tree, config.path);
  } else if (config.trinumber != 0) {
    InitRandom(&tree, config.trinumber);
  } else {
    printf("Usage (file mode): %s -f <filename>\n", bin_name);
    printf("or (random mode):  %s -t <triname>\n", bin_name);
    return 1;
  }

  Traverse(&tree, &config);
}
