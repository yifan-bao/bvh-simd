#include "bvh.h"

#define FREQUENCY 2.0e9  // disable turbo boost
// optimization used
// 1. SAH
// 2. blocking for imaging
// 3. binning
// TODO: SIMD, Data Layout, Numerical Issue
// TODO: Topology Optimization, Spatial Split

// bin count
#define BINS 8

// global variables
// TODO(xiaoyuan) init these variables
typedef struct {
  Tri *tri;
  uint *triIdx;
  BVHNode *bvhNode;
  uint rootNodeIdx;
  uint nodesUsed;
  unsigned long long N;
} BVHTree;

// forward declarations
void Subdivide(BVHTree *tree, uint nodeIdx);
void UpdateNodeBounds(BVHTree *tree, uint nodeIdx);


// functions
void IntersectTri(Ray *ray, const Tri *tri) {
  const float3 edge1 = SubFloat3(tri->vertex1, tri->vertex0);   // 3 flops
  const float3 edge2 = SubFloat3(tri->vertex2, tri->vertex0);   // 3 flops
  const float3 h = cross(ray->D, edge2);                        // 9 flops
  const float a = dot(edge1, h);                                // 5 flops
  if (a > -0.0001f && a < 0.0001f) {                            // 2 flops
#ifdef COUNTFLOPS
    flopcount += 22;
#endif
    return;
  }
  
  const float f = 1 / a;                              // 1 flop
  const float3 s = SubFloat3(ray->O, tri->vertex0);   // 3 flops
  const float u = f * dot(s, h);                      // 6 flops
  if (u < 0 || u > 1) {                               // 2 flops
#ifdef COUNTFLOPS
    flopcount += 34;
#endif
    return;
  }

  const float3 q = cross(s, edge1);     // 9 flops
  const float v = f * dot(ray->D, q);   // 6 flops
  if (v < 0 || u + v > 1) {             // 3 flops
#ifdef COUNTFLOPS
    flopcount += 52;
#endif
    return;
  }

  const float t = f * dot(edge2, q);    // 6 flops
  if (t > 0.0001f) {                    // 1 flop
    ray->t = min(ray->t, t);            // 1 flop
#ifdef COUNTFLOPS
    flopcount += 60;
#endif
    return;
  }

#ifdef COUNTFLOPS
  flopcount += 59;
#endif
}

inline float IntersectAABB(const Ray *ray, const float3 bmin, const float3 bmax) {
#ifdef COUNTFLOPS
  flopcount += 25;
#endif
  float tx1 = (bmin.x - ray->O.x) * ray->rD.x;
  float tx2 = (bmax.x - ray->O.x) * ray->rD.x;

  float tmin = min(tx1, tx2);
  float tmax = max(tx1, tx2);
  
  float ty1 = (bmin.y - ray->O.y) * ray->rD.y;
  float ty2 = (bmax.y - ray->O.y) * ray->rD.y;

  tmin = max(tmin, min(ty1, ty2));
  tmax = min(tmax, max(ty1, ty2));
  
  float tz1 = (bmin.z - ray->O.z) * ray->rD.z;
  float tz2 = (bmax.z - ray->O.z) * ray->rD.z;
  
  tmin = max(tmin, min(tz1, tz2));
  tmax = min(tmax, max(tz1, tz2));
  
  if (tmax >= tmin && tmin < ray->t && tmax > 0)
    return tmin;
  else
    return 1e30f;
}

void IntersectBVH(BVHTree *tree, Ray *ray) {
  BVHNode *node = tree->bvhNode + tree->rootNodeIdx;
  BVHNode *stack[64];
  uint stackPtr = 0;
  while (1) {
    if (BVHNodeIsLeaf(&node)) {
      for (uint i = 0; i < node->triCount; i++) {
        IntersectTri(ray, tree->tri + tree->triIdx[node->leftFirst + i]);
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
    float dist1 = IntersectAABB(ray, child1->aabbMin, child1->aabbMax);
    float dist2 = IntersectAABB(ray, child2->aabbMin, child2->aabbMax);
    if (dist1 > dist2) {
      swap(dist1, dist2);
      swap(child1, child2);
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
  // create the BVH node pool
  // bvhNode = (BVHNode*)_aligned_malloc( sizeof( BVHNode ) * N * 2, 64 );
  tree->bvhNode = (BVHNode*)aligned_alloc(64, sizeof(BVHNode) * tree->N * 2);

  // populate triangle index array
  for (int i = 0; i < tree->N; i++)
    tree->triIdx[i] = i;
  
  // calculate triangle centroids for partitioning
  for (int i = 0; i < tree->N; i++) {
    // Strength Reduced
    Tri *tri = tree->tri + i;
    tri->centroid = AddFloat3(AddFloat3(tri->vertex0, tri->vertex1), tri->vertex2);
    tri->centroid = MulConstFloat3(tri->centroid, 1.0f / 3.0f);
  }
  
  // assign all triangles to root node
  BVHNode *root = tree->bvhNode + tree->rootNodeIdx;
  root->leftFirst = 0;
  root->triCount = tree->N;

  UpdateNodeBounds(tree, tree->rootNodeIdx);
  // subdivide recursively
  Subdivide(tree, tree->rootNodeIdx);
  
  // TODO(xiaoyuan) time here
}

void UpdateNodeBounds(BVHTree *tree, uint nodeIdx) {
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
}

float FindBestSplitPlane(BVHTree *tree, BVHNode *node, int *axis, float *splitPos) {
  float bestCost = 1e30f;
  for (int a = 0; a < 3; a++) {
    float boundsMin = 1e30f, boundsMax = -1e30f;
    for (uint i = 0; i < node->triCount; i++) {
      Tri *triangle = tree->tri + tree->triIdx[node->leftFirst + i];
      boundsMin = min(boundsMin, float3_at(triangle->centroid, a));
      boundsMax = max(boundsMax, float3_at(triangle->centroid, a));
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
      int binIdx = min(BINS - 1, (int)((float3_at(triangle->centroid, a) - boundsMin) * scale));  // TODO: do we need the min function?
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

float CalculateNodeCost(BVHNode *node) {
  float3 e = SubFloat3(node->aabbMax, node->aabbMin);  // extent of the node
  float surfaceArea = e.x * e.y + e.y * e.z + e.z * e.x;
  return node->triCount * surfaceArea;
}

void Subdivide(BVHTree *tree, uint nodeIdx) {
  // terminate recursion
  BVHNode *node = tree->bvhNode + nodeIdx;
  // determine split axis using SAH
  int axis;
  float splitPos;
  float splitCost = FindBestSplitPlane(tree, node, &axis, &splitPos);
  // TODO: binary or wide BVH considering the problem of the wide BVH is that it may contain too many empty slots.
  float nosplitCost = CalculateNodeCost(node);
  if (splitCost >= nosplitCost) return;
  // in-place partition
  int i = node->leftFirst;
  int j = i + node->triCount - 1;
  while (i <= j) {
    if (float3_at(tree->tri[tree->triIdx[i]].centroid, axis) < splitPos)
      i++;
    else
      swap(tree->triIdx[i], tree->triIdx[j--]);
  }
  // abort split if one of the sides is empty
  int leftCount = i - node->leftFirst;
  if (leftCount == 0 || leftCount == node->triCount) return;
  // create child nodes
  int leftChildIdx = tree->nodesUsed++;
  int rightChildIdx = tree->nodesUsed++;
  // TODO: how to express the tree in array
  tree->bvhNode[leftChildIdx].leftFirst = node->leftFirst;
  tree->bvhNode[leftChildIdx].triCount = leftCount;
  tree->bvhNode[rightChildIdx].leftFirst = i;
  tree->bvhNode[rightChildIdx].triCount = node->triCount - leftCount;
  node->leftFirst = leftChildIdx;
  node->triCount = 0;
  UpdateNodeBounds(tree, leftChildIdx);
  UpdateNodeBounds(tree, rightChildIdx);
  // recurse
  Subdivide(tree, leftChildIdx);
  Subdivide(tree, rightChildIdx);
}

void InitRandom(BVHTree* tree, int triCount) {
  // malloc
  tree->N = triCount;
  tree->tri = malloc(tree->N * sizeof(Tri));
  tree->triIdx = malloc(tree->N * sizeof(uint));
  // intialize a scene with N random triangles
  for (int i = 0; i < tree->N; i++) {
    float3 r0 = make_float3_3(RandomFloat(), RandomFloat(), RandomFloat());
    float3 r1 = make_float3_3(RandomFloat(), RandomFloat(), RandomFloat());
    float3 r2 = make_float3_3(RandomFloat(), RandomFloat(), RandomFloat());
    tree->tri[i].vertex0 = SubFloat3(MulConstFloat3(r0, 9), make_float3(5));
    tree->tri[i].vertex1 = AddFloat3(tree->tri[i].vertex0, r1);
    tree->tri[i].vertex2 = AddFloat3(tree->tri[i].vertex0, r2);
  }
  BuildBVH(tree);
}

void Init(BVHTree *tree, char* filename, int triCount) {
  tree->N = triCount;
  tree->tri = malloc(tree->N * sizeof(Tri));
  tree->tri[tree->N - 1].vertex0.x = 0;
  tree->triIdx = malloc(tree->N * sizeof(uint));
  
  int t;
  int num = 9;
  FILE* file = fopen(filename, "r");
  for (t = 0; t < triCount && num == 9; t++) {
    num = fscanf(file, "%f %f %f %f %f %f %f %f %f\n", &tree->tri[t].vertex0.x, &tree->tri[t].vertex0.y, &tree->tri[t].vertex0.z,
                 &tree->tri[t].vertex1.x, &tree->tri[t].vertex1.y, &tree->tri[t].vertex1.z, &tree->tri[t].vertex2.x, &tree->tri[t].vertex2.y,
                 &tree->tri[t].vertex2.z);
    if (num != 9) {
      printf("file format wrong\n");
      exit(1);
    }
  }
  fclose(file);
  // construct the BVH
  BuildBVH(tree);
}

void Tick(BVHTree *tree) {
  float3 p0 = make_float3_3(-1, 1, 2);
  float3 p1 = make_float3_3(1, 1, 2);
  float3 p2 = make_float3_3(-1, -1, 2);

  Ray ray;
  RayInit(&ray);
  for (int y = 0; y < SCRHEIGHT; y += 4) {
    for (int x = 0; x < SCRWIDTH; x += 4) {
      // tiling/blocking
      for (int v = 0; v < 4; v++)
        for (int u = 0; u < 4; u++) {
          ray.O = make_float3_3(-1.5f, -0.2f, -2.5f);
          // float3 pixelPos = ray.O + p0 + (p1 - p0) * ((x + u) / (float)SCRWIDTH) + (p2 - p0) * ((y + v) / (float)SCRHEIGHT);
          float3 pixelPos = AddFloat3(ray.O, p0);
          pixelPos = AddFloat3(pixelPos, MulConstFloat3(SubFloat3(p1, p0), (x + u) / (float)SCRWIDTH));
          pixelPos = AddFloat3(pixelPos, MulConstFloat3(SubFloat3(p2, p0), (y + v) / (float)SCRHEIGHT));

          ray.D = normalize(SubFloat3(pixelPos, ray.O));
          ray.t = 1e30f;
          ray.rD = make_float3_3(1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z);
          IntersectBVH(tree, &ray);
        }
    }
  }
  // TODO(xiaoyuan): time here
  // printf("tracing cycles: %f\n", cycles );
  // printf("tracing cycles: %f\n", elapsed / 1000 * FREQUENCY);
  // printf("tracing time: %.2fms (%5.2fK rays/s)\n", elapsed, sqr(630) / elapsed);
#ifdef COUNTFLOPS
  printf("FLOPS COUNT: %lld flops\n", flopcount);
#endif
}

int main(int argc, char* argv[]) {
  BVHTree tree = {
    .N = 0,
    .bvhNode = NULL,
    .tri = NULL,
    .triIdx = NULL,
    .rootNodeIdx = 0,
    .nodesUsed = 2,
  };

  if (argc == 2)
    InitRandom(&tree, atoi(argv[1]));
  else if (argc == 3)
    Init(&tree, argv[1], atoi(argv[2]));
  else
    return 1;
  
  Tick(&tree);
}
