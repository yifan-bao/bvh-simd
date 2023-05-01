#include "precomp.h"
#include "quickbuild.h"


TheApp* CreateApp() { return new QuickBuildApp(); }

// triangle count
#define N	30000 // hardcoded for the unity vehicle mesh

// bin count
#define BINS 8

// forward declarations
void Subdivide( uint nodeIdx );
void UpdateNodeBounds( uint nodeIdx );

// minimal structs
struct Tri { float3 vertex0, vertex1, vertex2; float3 centroid; };
struct BVHNode
{
    float3 aabbMin, aabbMax;
	uint leftFirst, triCount;
	bool isLeaf() { return triCount > 0; }
};
struct aabb
{
	float3 bmin = 1e30f, bmax = -1e30f;
	void grow( float3 p ) { bmin = fminf( bmin, p ); bmax = fmaxf( bmax, p ); }
	void grow( aabb& b ) { if (b.bmin.x != 1e30f) { grow( b.bmin ); grow( b.bmax ); } }
	float area()
	{
		float3 e = bmax - bmin; // box extent
		return e.x * e.y + e.y * e.z + e.z * e.x;
	}
};
struct Bin { aabb bounds; int triCount = 0; };
__declspec(align(64)) struct Ray
{
	Ray() { O = D = rD = float3(1.0); }
    float3 O; float dummy1;
    float3 D; float dummy2;
    float3 rD; float dummy3;
    float t = 1e30f;
};

// application data
Tri tri[N];
uint triIdx[N];
BVHNode* bvhNode = 0;
uint rootNodeIdx = 0, nodesUsed = 2;

// functions

void IntersectTri( Ray& ray, const Tri& tri )
{
	const float3 edge1 = tri.vertex1 - tri.vertex0;	// 3 flops
	const float3 edge2 = tri.vertex2 - tri.vertex0; // 3 flops
	const float3 h = cross( ray.D, edge2 );			// 9 flops
	const float a = dot( edge1, h );				// 5 flops
	if (a > -0.0001f && a < 0.0001f) {				// 2 flops
		#ifdef COUNTFLOPS
			flopcount += 22;
		#endif
		return ;
	}
	const float f = 1 / a;							// 1 flop
	const float3 s = ray.O - tri.vertex0;			// 3 flops
	const float u = f * dot( s, h );				// 6 flops
	if (u < 0 || u > 1) {							// 2 flops
		#ifdef COUNTFLOPS
			flopcount += 34;
		#endif
		return;
	}
	const float3 q = cross( s, edge1 );				// 9 flops
	const float v = f * dot( ray.D, q );			// 6 flops
	if (v < 0 || u + v > 1) {						// 3 flops
		#ifdef COUNTFLOPS
			flopcount += 52;
		#endif
		return;
	}
	const float t = f * dot( edge2, q );			// 6 flops
	if (t > 0.0001f) {									// 1 flop
		ray.t = min( ray.t, t );						// 1 flop
		#ifdef COUNTFLOPS
			flopcount += 60;
		#endif
		return;
	}
	#ifdef COUNTFLOPS
		flopcount += 59;
	#endif
}

inline float IntersectAABB( const Ray& ray, const float3 bmin, const float3 bmax )
{
	#ifdef COUNTFLOPS
		flopcount += 25;
	#endif
	float tx1 = (bmin.x - ray.O.x) * ray.rD.x, tx2 = (bmax.x - ray.O.x) * ray.rD.x;
	float tmin = min( tx1, tx2 ), tmax = max( tx1, tx2 );
	float ty1 = (bmin.y - ray.O.y) * ray.rD.y, ty2 = (bmax.y - ray.O.y) * ray.rD.y;
	tmin = max( tmin, min( ty1, ty2 ) ), tmax = min( tmax, max( ty1, ty2 ) );
	float tz1 = (bmin.z - ray.O.z) * ray.rD.z, tz2 = (bmax.z - ray.O.z) * ray.rD.z;
	tmin = max( tmin, min( tz1, tz2 ) ), tmax = min( tmax, max( tz1, tz2 ) );
	if (tmax >= tmin && tmin < ray.t && tmax > 0) return tmin; else return 1e30f;
}


void IntersectBVH( Ray& ray )
{
	BVHNode* node = &bvhNode[rootNodeIdx], * stack[64];
	uint stackPtr = 0;
	while (1)
	{
		if (node->isLeaf())
		{
			for (uint i = 0; i < node->triCount; i++)
				IntersectTri( ray, tri[triIdx[node->leftFirst + i]] );
			if (stackPtr == 0) break; else node = stack[--stackPtr];
			continue;
		}
		BVHNode* child1 = &bvhNode[node->leftFirst];
		BVHNode* child2 = &bvhNode[node->leftFirst + 1];
		float dist1 = IntersectAABB( ray, child1->aabbMin, child1->aabbMax );
		float dist2 = IntersectAABB( ray, child2->aabbMin, child2->aabbMax );
		if (dist1 > dist2) { 
			#ifdef COUNTFLOPS
				flopcount += 25;
			#endif
			swap( dist1, dist2 ); swap( child1, child2 ); 
		}
		if (dist1 == 1e30f)
		{
			if (stackPtr == 0) break; else node = stack[--stackPtr];
		}
		else
		{
			node = child1;
			if (dist2 != 1e30f) stack[stackPtr++] = child2;
		}
	}
}

void BuildBVH()
{
	// create the BVH node pool
	// bvhNode = (BVHNode*)_aligned_malloc( sizeof( BVHNode ) * N * 2, 64 );
    bvhNode = (BVHNode*)aligned_alloc(64, sizeof(BVHNode) * N * 2);

	// populate triangle index array
	for (int i = 0; i < N; i++) triIdx[i] = i;
	// calculate triangle centroids for partitioning
	for (int i = 0; i < N; i++)
		tri[i].centroid = (tri[i].vertex0 + tri[i].vertex1 + tri[i].vertex2) * 0.3333f;
	// assign all triangles to root node
	BVHNode& root = bvhNode[rootNodeIdx];
	root.leftFirst = 0, root.triCount = N;
	UpdateNodeBounds( rootNodeIdx );
	// subdivide recursively
	Timer t;
	Subdivide( rootNodeIdx );
	printf( "BVH (%i nodes) constructed in %.2fms.\n", nodesUsed, t.elapsed() * 1000 );
}

void UpdateNodeBounds( uint nodeIdx )
{
	BVHNode& node = bvhNode[nodeIdx];
	node.aabbMin = float3( 1e30f );
	node.aabbMax = float3( -1e30f );
	for (uint first = node.leftFirst, i = 0; i < node.triCount; i++)
	{
		uint leafTriIdx = triIdx[first + i];
		Tri& leafTri = tri[leafTriIdx];
		node.aabbMin = fminf( node.aabbMin, leafTri.vertex0 );
		node.aabbMin = fminf( node.aabbMin, leafTri.vertex1 );
		node.aabbMin = fminf( node.aabbMin, leafTri.vertex2 );
		node.aabbMax = fmaxf( node.aabbMax, leafTri.vertex0 );
		node.aabbMax = fmaxf( node.aabbMax, leafTri.vertex1 );
		node.aabbMax = fmaxf( node.aabbMax, leafTri.vertex2 );
	}
}

float FindBestSplitPlane( BVHNode& node, int& axis, float& splitPos )
{
	float bestCost = 1e30f;
	for (int a = 0; a < 3; a++)
	{
		float boundsMin = 1e30f, boundsMax = -1e30f;
		for (uint i = 0; i < node.triCount; i++)
		{
			Tri& triangle = tri[triIdx[node.leftFirst + i]];
			boundsMin = min( boundsMin, triangle.centroid[a] );
			boundsMax = max( boundsMax, triangle.centroid[a] );
		}
		if (boundsMin == boundsMax) continue;
		// populate the bins
		Bin bin[BINS];
		float scale = BINS / (boundsMax - boundsMin);
		for (uint i = 0; i < node.triCount; i++)
		{
			Tri& triangle = tri[triIdx[node.leftFirst + i]];
			int binIdx = min( BINS - 1, (int)((triangle.centroid[a] - boundsMin) * scale) );
			bin[binIdx].triCount++;
			bin[binIdx].bounds.grow( triangle.vertex0 );
			bin[binIdx].bounds.grow( triangle.vertex1 );
			bin[binIdx].bounds.grow( triangle.vertex2 );
		}
		// gather data for the 7 planes between the 8 bins
		float leftArea[BINS - 1], rightArea[BINS - 1];
		int leftCount[BINS - 1], rightCount[BINS - 1];
		aabb leftBox, rightBox;
		int leftSum = 0, rightSum = 0;
		for (int i = 0; i < BINS - 1; i++)
		{
			leftSum += bin[i].triCount;
			leftCount[i] = leftSum;
			leftBox.grow( bin[i].bounds );
			leftArea[i] = leftBox.area();
			rightSum += bin[BINS - 1 - i].triCount;
			rightCount[BINS - 2 - i] = rightSum;
			rightBox.grow( bin[BINS - 1 - i].bounds );
			rightArea[BINS - 2 - i] = rightBox.area();
		}
		// calculate SAH cost for the 7 planes
		scale = (boundsMax - boundsMin) / BINS;
		for (int i = 0; i < BINS - 1; i++)
		{
			float planeCost = leftCount[i] * leftArea[i] + rightCount[i] * rightArea[i];
			if (planeCost < bestCost)
				axis = a, splitPos = boundsMin + scale * (i + 1), bestCost = planeCost;
		}
	}
	return bestCost;
}

float CalculateNodeCost( BVHNode& node )
{
	float3 e = node.aabbMax - node.aabbMin; // extent of the node
	float surfaceArea = e.x * e.y + e.y * e.z + e.z * e.x;
	return node.triCount * surfaceArea;
}

void Subdivide( uint nodeIdx )
{
	// terminate recursion
	BVHNode& node = bvhNode[nodeIdx];
	// determine split axis using SAH
	int axis;
	float splitPos;
	float splitCost = FindBestSplitPlane( node, axis, splitPos );
	float nosplitCost = CalculateNodeCost( node );
	if (splitCost >= nosplitCost) return;
	// in-place partition
	int i = node.leftFirst;
	int j = i + node.triCount - 1;
	while (i <= j)
	{
		if (tri[triIdx[i]].centroid[axis] < splitPos)
			i++;
		else
			swap( triIdx[i], triIdx[j--] );
	}
	// abort split if one of the sides is empty
	int leftCount = i - node.leftFirst;
	if (leftCount == 0 || leftCount == node.triCount) return;
	// create child nodes
	int leftChildIdx = nodesUsed++;
	int rightChildIdx = nodesUsed++;
	bvhNode[leftChildIdx].leftFirst = node.leftFirst;
	bvhNode[leftChildIdx].triCount = leftCount;
	bvhNode[rightChildIdx].leftFirst = i;
	bvhNode[rightChildIdx].triCount = node.triCount - leftCount;
	node.leftFirst = leftChildIdx;
	node.triCount = 0;
	UpdateNodeBounds( leftChildIdx );
	UpdateNodeBounds( rightChildIdx );
	// recurse
	Subdivide( leftChildIdx );
	Subdivide( rightChildIdx );
}

void QuickBuildApp::Init()
{
	FILE* file = fopen( "assets/bigben.tri", "r" );
	for (int t = 0; t < N; t++)
		fscanf( file, "%f %f %f %f %f %f %f %f %f\n",
			&tri[t].vertex0.x, &tri[t].vertex0.y, &tri[t].vertex0.z,
			&tri[t].vertex1.x, &tri[t].vertex1.y, &tri[t].vertex1.z,
			&tri[t].vertex2.x, &tri[t].vertex2.y, &tri[t].vertex2.z );
	BuildBVH();
}

void QuickBuildApp::Tick( float deltaTime )
{
	// draw the scene
	// screen->Clear( 0 );
	float3 p0( -1, 1, 2 ), p1( 1, 1, 2 ), p2( -1, -1, 2 );
	Ray ray;
	Timer t;
	for (int y = 0; y < SCRHEIGHT; y += 4) for (int x = 0; x < SCRWIDTH; x += 4)
	{
		for (int v = 0; v < 4; v++) for (int u = 0; u < 4; u++)
		{
			ray.O = float3( -1.5f, -0.2f, -2.5f );
			float3 pixelPos = ray.O + p0 + (p1 - p0) * ((x + u) / (float)SCRWIDTH) + (p2 - p0) * ((y + v) / (float)SCRHEIGHT);
			ray.D = normalize( pixelPos - ray.O ), ray.t = 1e30f;
			ray.rD = float3( 1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z );
			IntersectBVH( ray );
			
		}
	}
	float elapsed = t.elapsed() * 1000;
	printf( "tracing time: %.2fms (%5.2fK rays/s)\n", elapsed, sqr( 630 ) / elapsed );
	#ifdef COUNTFLOPS
	printf("FLOPS COUNT: %lld flops\n", flopcount);
	#endif
}

static TheApp* app = 0;
int main(void) {
    app = CreateApp();
    app->Init();
    app->Tick(1);
}