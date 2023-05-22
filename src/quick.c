#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  Tri *tri;
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
    ray->t = fmin(ray->t, t);            // 1 flop
#ifdef COUNTFLOPS
    flopcount += 60;
#endif
    return;
  }

#ifdef COUNTFLOPS
  flopcount += 59;
#endif
}

float IntersectAABB(const Ray *ray, const float3 bmin, const float3 bmax) {
#ifdef COUNTFLOPS
  flopcount += 25;
#endif
  float tx1 = (bmin.x - ray->O.x) * ray->rD.x;
  float tx2 = (bmax.x - ray->O.x) * ray->rD.x;

  float tmin = fmin(tx1, tx2);
  float tmax = fmax(tx1, tx2);
  
  float ty1 = (bmin.y - ray->O.y) * ray->rD.y;
  float ty2 = (bmax.y - ray->O.y) * ray->rD.y;

  tmin = fmax(tmin, fmin(ty1, ty2));
  tmax = fmin(tmax, fmax(ty1, ty2));
  
  float tz1 = (bmin.z - ray->O.z) * ray->rD.z;
  float tz2 = (bmax.z - ray->O.z) * ray->rD.z;
  
  tmin = fmax(tmin, fmin(tz1, tz2));
  tmax = fmin(tmax, fmax(tz1, tz2));
  
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
    if (BVHNodeIsLeaf(node)) {
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
  printf("before build flops: %lld\n", flopcount);
#endif

  // time
  myInt64 build_start, build_end;
  build_start = start_tsc();

  // create the BVH node pool
  // bvhNode = (BVHNode*)_aligned_malloc( sizeof( BVHNode ) * N * 2, 64 );
  tree->bvhNode = (BVHNode*)aligned_alloc(64, sizeof(BVHNode) * tree->N * 2);

  // populate triangle index array
  for (ull i = 0; i < tree->N; i++)
    tree->triIdx[i] = i;
  
  // calculate triangle centroids for partitioning
  for (ull i = 0; i < tree->N; i++) {
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
  build_end = stop_tsc(build_start);
  double cycles = (double) build_end;
  printf("build cycles: %f\n", cycles);
#ifdef COUNTFLOPS
  printf("build flops: %lld\n", flopcount);
#endif
  printf("BVH tree built with %d nodes\n", tree->nodesUsed);
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
  uint i = node->leftFirst;
  uint j = i + node->triCount - 1;
  while (i <= j) {
    if (float3_at(tree->tri[tree->triIdx[i]].centroid, axis) < splitPos)
      i++;
    else
      swap_uint(tree->triIdx + i, tree->triIdx + j--);
  }
  // abort split if one of the sides is empty
  uint leftCount = i - node->leftFirst;
  if (leftCount == 0 || leftCount == node->triCount) return;
  // create child nodes
  uint leftChildIdx = tree->nodesUsed++;
  uint rightChildIdx = tree->nodesUsed++;
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
  int triCount = 0;  // Initialize triangle count

  // Count the number of triangles in the file
  FILE* countFile = fopen(filename, "r");
  if (countFile == NULL) {
    printf("Failed to open file: %s\n", filename);
    exit(1);
  }

  float dummy;
  while (fscanf(countFile, "%f %f %f %f %f %f %f %f %f\n", &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy) == 9) {
    triCount++;
  }
  fclose(countFile);

  printf("Number of triangles: %d in file %s\n", triCount, filename);

  // Allocate memory for triangles and triangle indices
  tree->N = triCount;
  tree->tri = malloc(tree->N * sizeof(Tri));
  tree->triIdx = malloc(tree->N * sizeof(uint));

  // Read triangle data from file
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    printf("Failed to open file: %s\n", filename);
    exit(1);
  }

  for (int t = 0; t < triCount; t++) {
    if (fscanf(file, "%f %f %f %f %f %f %f %f %f\n", &tree->tri[t].vertex0.x, &tree->tri[t].vertex0.y, &tree->tri[t].vertex0.z,
               &tree->tri[t].vertex1.x, &tree->tri[t].vertex1.y, &tree->tri[t].vertex1.z, &tree->tri[t].vertex2.x, &tree->tri[t].vertex2.y,
               &tree->tri[t].vertex2.z) != 9) {
      printf("File format is incorrect\n");
      exit(1);
    }
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
  printf("before traverse flops: %lld\n", flopcount);
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
  double cycles = (double) traverse_end;
  printf("traverse cycles: %f\n", cycles);
#ifdef COUNTFLOPS
  printf("FLOPS COUNT: %lld flops\n", flopcount);
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
