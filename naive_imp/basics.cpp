#include "precomp.h"
#include "basics.h"
#define FREQUENCY 2.0e9 // disable turbo boost

static uint seed = 0x12345678;
uint RandomUInt()
{
	seed ^= seed << 13;
	seed ^= seed >> 17;
	seed ^= seed << 5;
	return seed;
}
uint RandomUInt( uint& seed )
{
	seed ^= seed << 13;
	seed ^= seed >> 17;
	seed ^= seed << 5;
	return seed;
}
float Rand( float range ) { return RandomFloat() * range; }
TheApp* CreateApp() { return new BasicBVHApp(); }
float RandomFloat() { return RandomUInt() * 2.3283064365387e-10f; }
float RandomFloat( uint& seed ) { return RandomUInt( seed ) * 2.3283064365387e-10f; }

// triangle count
#define N	50000 // set it larger to see the effect of the BVH

// forward declarations
void Subdivide( uint nodeIdx );
void UpdateNodeBounds( uint nodeIdx );

// minimal structs
struct Tri { float3 vertex0, vertex1, vertex2; float3 centroid; };
// __declspec(align(32)) 
struct BVHNode
{
	float3 aabbMin, aabbMax;
	uint leftFirst, triCount;
	bool isLeaf() { return triCount > 0; }
};
struct Ray { float3 O, D; float t = 1e30f; };

// application data
Tri tri[N];
uint triIdx[N];
BVHNode bvhNode[N * 2];
uint rootNodeIdx = 0, nodesUsed = 1;

// functions

void IntersectTri( Ray& ray, const Tri& tri )
{
	const float3 edge1 = tri.vertex1 - tri.vertex0;		// 3 flops
	const float3 edge2 = tri.vertex2 - tri.vertex0; 	// 3 flops
	const float3 h = cross( ray.D, edge2 );				// 9 flops
	const float a = dot( edge1, h );					// 5 flops
	if (a > -0.0001f && a < 0.0001f) {					// 2 flops
		#ifdef COUNTFLOPS
			flopcount += 22;
		#endif
		return; 		// ray parallel to triangle		
	}
	const float f = 1 / a;								// 1 flop
	const float3 s = ray.O - tri.vertex0;				// 3 flops
	const float u = f * dot( s, h );					// 6 flops
	if (u < 0 || u > 1) {								// 2 flops
		#ifdef COUNTFLOPS
			flopcount += 34;
		#endif
		return;
	}
	const float3 q = cross( s, edge1 );					// 9 flops
	const float v = f * dot( ray.D, q );				// 6 flops
	if (v < 0 || u + v > 1) {							// 3 flops
		#ifdef COUNTFLOPS
			flopcount += 52;
		#endif
		return;
	}
	const float t = f * dot( edge2, q );				// 6 flops
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

// flopcount = 25
bool IntersectAABB( const Ray& ray, const float3 bmin, const float3 bmax )
{
	#ifdef COUNTFLOPS
		flopcount += 25;
	#endif
	float tx1 = (bmin.x - ray.O.x) / ray.D.x, tx2 = (bmax.x - ray.O.x) / ray.D.x;
	float tmin = min( tx1, tx2 ), tmax = max( tx1, tx2 );
	float ty1 = (bmin.y - ray.O.y) / ray.D.y, ty2 = (bmax.y - ray.O.y) / ray.D.y;
	tmin = max( tmin, min( ty1, ty2 ) ), tmax = min( tmax, max( ty1, ty2 ) );
	float tz1 = (bmin.z - ray.O.z) / ray.D.z, tz2 = (bmax.z - ray.O.z) / ray.D.z;
	tmin = max( tmin, min( tz1, tz2 ) ), tmax = min( tmax, max( tz1, tz2 ) );
	return tmax >= tmin && tmin < ray.t && tmax > 0;
}

void IntersectBVH( Ray& ray, const uint nodeIdx )
{
	BVHNode& node = bvhNode[nodeIdx];
	if (!IntersectAABB( ray, node.aabbMin, node.aabbMax )) return;
	if (node.isLeaf())
	{
		for (uint i = 0; i < node.triCount; i++ )
			IntersectTri( ray, tri[triIdx[node.leftFirst + i]] );
	}
	else
	{
		IntersectBVH( ray, node.leftFirst );
		IntersectBVH( ray, node.leftFirst + 1 );
	}
}

void BuildBVH()
{
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
		node.aabbMin = fminf( node.aabbMin, leafTri.vertex0 ),
		node.aabbMin = fminf( node.aabbMin, leafTri.vertex1 ),
		node.aabbMin = fminf( node.aabbMin, leafTri.vertex2 ),
		node.aabbMax = fmaxf( node.aabbMax, leafTri.vertex0 ),
		node.aabbMax = fmaxf( node.aabbMax, leafTri.vertex1 ),
		node.aabbMax = fmaxf( node.aabbMax, leafTri.vertex2 );
	}
}

void Subdivide( uint nodeIdx )
{
	// terminate recursion
	BVHNode& node = bvhNode[nodeIdx];
	if (node.triCount <= 2) return;
	// determine split axis and position
	float3 extent = node.aabbMax - node.aabbMin;
	int axis = 0;
	if (extent.y > extent.x) axis = 1;
	if (extent.z > extent[axis]) axis = 2;
	float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;
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

void BasicBVHApp::Init()
{
	
	FILE* file = fopen( "assets/dragon.tri", "r" );
	float a, b, c, d, e, f, g, h, i;
	for (int t = 0; t < N; t++)
	{
		fscanf( file, "%f %f %f %f %f %f %f %f %f\n",
			&a, &b, &c, &d, &e, &f, &g, &h, &i );
		tri[t].vertex0 = float3( a, b, c );
		tri[t].vertex1 = float3( d, e, f );
		tri[t].vertex2 = float3( g, h, i );
	}
	fclose( file );
	// construct the BVH
	BuildBVH();
}

void BasicBVHApp::Tick( float deltaTime )
{
	// draw the scene
	// screen->Clear( 0 );
	float3 p0( -1, 1, -15 ), p1( 1, 1, -15 ), p2( -1, -1, -15 );
	Ray ray;
	Timer t;
	for (int y = 0; y < SCRHEIGHT; y++) for (int x = 0; x < SCRWIDTH; x++)
	{
		float3 pixelPos = p0 + (p1 - p0) * (x / (float)SCRWIDTH) + (p2 - p0) * (y / (float)SCRHEIGHT);
		ray.O = float3( 0, 0, -18 );
		ray.D = normalize( pixelPos - ray.O );
		ray.t = 1e30f;

		IntersectBVH( ray, rootNodeIdx );

		// if (ray.t < 1e30f) screen->Plot( x, y, 0xffffff );
	}
	float elapsed = t.elapsed() * 1000;
	printf("tracing cycles: %f\n", elapsed/1000 * FREQUENCY );
	printf( "tracing time: %.2fms (%5.2fK rays/s)\n", elapsed, sqr( 630 ) / elapsed );
	#ifdef COUNTFLOPS
		printf("FLOPS COUNT: %lld flops\n", flopcount);
	#endif
}


// find the app implementation
static TheApp* app = 0;
TheApp* CreateApp();
int main(void) {
	app = CreateApp();
	// app->screen = screen;
	app->Init();
	app->Tick(1);
}
// EOF