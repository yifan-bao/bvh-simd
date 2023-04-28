#include <chrono>
#include <fstream>
#include <vector>
#include <list>
#include <string>
#include <thread>
#include <math.h>
#include <algorithm>
#include <assert.h>
#include <sys/uio.h>

#include "lib/stb_image.h"
#include <immintrin.h>

// header for AVX, and every technology before it.
// if your CPU does not support this (unlikely), include the appropriate header instead.
// see: https://stackoverflow.com/a/11228864/2844473
#include <immintrin.h>
using namespace std;

// basic types
typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned short ushort;

namespace Tmpl8
{

};

using namespace Tmpl8;
// aligned memory allocations
#ifdef _MSC_VER
#define ALIGN( x ) __declspec( align( x ) )
#define MALLOC64( x ) ( ( x ) == 0 ? 0 : _aligned_malloc( ( x ), 64 ) )
#define FREE64( x ) _aligned_free( x )
#else
#define ALIGN( x ) __attribute__( ( aligned( x ) ) )
#define MALLOC64( x ) ( ( x ) == 0 ? 0 : aligned_alloc( 64, ( x ) ) )
#define FREE64( x ) free( x )
#endif
#if defined(__GNUC__) && (__GNUC__ >= 4)
#define CHECK_RESULT __attribute__ ((warn_unused_result))
#elif defined(_MSC_VER) && (_MSC_VER >= 1700)
#define CHECK_RESULT _Check_return_
#else
#define CHECK_RESULT
#endif

// vector type placeholders, carefully matching OpenCL's layout and alignment
struct ALIGN( 8 ) int2
{
	int2() = default;
	int2( const int a, const int b ) : x( a ), y( b ) {}
	int2( const int a ) : x( a ), y( a ) {}
	union { struct { int x, y; }; int cell[2]; };
	int& operator [] ( const int n ) { return cell[n]; }
};
struct ALIGN( 8 ) uint2
{
	uint2() = default;
	uint2( const int a, const int b ) : x( a ), y( b ) {}
	uint2( const uint a ) : x( a ), y( a ) {}
	union { struct { uint x, y; }; uint cell[2]; };
	uint& operator [] ( const int n ) { return cell[n]; }
};
struct ALIGN( 8 ) float2
{
	float2() = default;
	float2( const float a, const float b ) : x( a ), y( b ) {}
	float2( const float a ) : x( a ), y( a ) {}
	union { struct { float x, y; }; float cell[2]; };
	float& operator [] ( const int n ) { return cell[n]; }
};
struct int3;
struct ALIGN( 16 ) int4
{
	int4() = default;
	int4( const int a, const int b, const int c, const int d ) : x( a ), y( b ), z( c ), w( d ) {}
	int4( const int a ) : x( a ), y( a ), z( a ), w( a ) {}
	int4( const int3 & a, const int d );
	union { struct { int x, y, z, w; }; int cell[4]; };
	int& operator [] ( const int n ) { return cell[n]; }
};
struct ALIGN( 16 ) int3
{
	int3() = default;
	int3( const int a, const int b, const int c ) : x( a ), y( b ), z( c ) {}
	int3( const int a ) : x( a ), y( a ), z( a ) {}
	int3( const int4 a ) : x( a.x ), y( a.y ), z( a.z ) {}
	union { struct { int x, y, z; int dummy; }; int cell[4]; };
	int& operator [] ( const int n ) { return cell[n]; }
};
struct uint3;
struct ALIGN( 16 ) uint4
{
	uint4() = default;
	uint4( const uint a, const uint b, const uint c, const uint d ) : x( a ), y( b ), z( c ), w( d ) {}
	uint4( const uint a ) : x( a ), y( a ), z( a ), w( a ) {}
	uint4( const uint3 & a, const uint d );
	union { struct { uint x, y, z, w; }; uint cell[4]; };
	uint& operator [] ( const int n ) { return cell[n]; }
};
struct ALIGN( 16 ) uint3
{
	uint3() = default;
	uint3( const uint a, const uint b, const uint c ) : x( a ), y( b ), z( c ) {}
	uint3( const uint a ) : x( a ), y( a ), z( a ) {}
	uint3( const uint4 a ) : x( a.x ), y( a.y ), z( a.z ) {}
	union { struct { uint x, y, z; uint dummy; }; uint cell[4]; };
	uint& operator [] ( const int n ) { return cell[n]; }
};
struct float3;
struct ALIGN( 16 ) float4
{
	float4() = default;
	float4( const float a, const float b, const float c, const float d ) : x( a ), y( b ), z( c ), w( d ) {}
	float4( const float a ) : x( a ), y( a ), z( a ), w( a ) {}
	float4( const float3 & a, const float d );
	float4( const float3 & a );
	union { struct { float x, y, z, w; }; float cell[4]; };
	float& operator [] ( const int n ) { return cell[n]; }
};
struct float3
{
	float3() = default;
	float3( const float a, const float b, const float c ) : x( a ), y( b ), z( c ) {}
	float3( const float a ) : x( a ), y( a ), z( a ) {}
	float3( const float4 a ) : x( a.x ), y( a.y ), z( a.z ) {}
	float3( const uint3 a ) : x( (float)a.x ), y( (float)a.y ), z( (float)a.z ) {}
	union { struct { float x, y, z; }; float cell[3]; };
	float& operator [] ( const int n ) { return cell[n]; }
};
struct ALIGN( 4 ) uchar4
{
	uchar4() = default;
	uchar4( const uchar a, const uchar b, const uchar c, const uchar d ) : x( a ), y( b ), z( c ), w( d ) {}
	uchar4( const uchar a ) : x( a ), y( a ), z( a ), w( a ) {}
	union { struct { uchar x, y, z, w; }; uchar cell[4]; };
	uchar& operator [] ( const int n ) { return cell[n]; }
};

#define FATALERROR( fmt, ... ) FatalError( "Error on line %d of %s: " fmt "\n", __LINE__, __FILE__, ##__VA_ARGS__ )
#define FATALERROR_IF( condition, fmt, ... ) do { if ( ( condition ) ) FATALERROR( fmt, ##__VA_ARGS__ ); } while ( 0 )
#define FATALERROR_IN( prefix, errstr, fmt, ... ) FatalError( prefix " returned error '%s' at %s:%d" fmt "\n", errstr, __FILE__, __LINE__, ##__VA_ARGS__ );
#define FATALERROR_IN_CALL( stmt, error_parser, fmt, ... ) do { auto ret = ( stmt ); if ( ret ) FATALERROR_IN( #stmt, error_parser( ret ), fmt, ##__VA_ARGS__ ) } while ( 0 )


// timer
struct Timer
{
	Timer() { reset(); }
	float elapsed() const
	{
		chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
		chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double> >(t2 - start);
		return (float)time_span.count();
	}
	void reset() { start = chrono::high_resolution_clock::now(); }
	chrono::high_resolution_clock::time_point start;
};

// swap
template <class T> void Swap( T& x, T& y ) { T t; t = x, x = y, y = t; }

// random numbers
uint RandomUInt();
uint RandomUInt( uint& seed );
float RandomFloat();
float RandomFloat( uint& seed );
float Rand( float range );

// Perlin noise
float noise2D( const float x, const float y );

// forward declaration of helper functions
void FatalError( const char* fmt, ... );
bool FileIsNewer( const char* file1, const char* file2 );
bool FileExists( const char* f );
bool RemoveFile( const char* f );
string TextFileRead( const char* _File );
int LineCount( const string s );
void TextFileWrite( const string& text, const char* _File );

// math
inline float fminf( float a, float b ) { return a < b ? a : b; }
inline float fmaxf( float a, float b ) { return a > b ? a : b; }
inline float rsqrtf( float x ) { return 1.0f / sqrtf( x ); }
inline float sqrf( float x ) { return x * x; }
inline int sqr( int x ) { return x * x; }

inline float2 make_float2( const float a, float b ) { float2 f2; f2.x = a, f2.y = b; return f2; }
inline float2 make_float2( const float s ) { return make_float2( s, s ); }
inline float2 make_float2( const float3& a ) { return make_float2( a.x, a.y ); }
inline float2 make_float2( const int2& a ) { return make_float2( float( a.x ), float( a.y ) ); } // explicit casts prevent gcc warnings
inline float2 make_float2( const uint2& a ) { return make_float2( float( a.x ), float( a.y ) ); }
inline int2 make_int2( const int a, const int b ) { int2 i2; i2.x = a, i2.y = b; return i2; }
inline int2 make_int2( const int s ) { return make_int2( s, s ); }
inline int2 make_int2( const int3& a ) { return make_int2( a.x, a.y ); }
inline int2 make_int2( const uint2& a ) { return make_int2( int( a.x ), int( a.y ) ); }
inline int2 make_int2( const float2& a ) { return make_int2( int( a.x ), int( a.y ) ); }
inline uint2 make_uint2( const uint a, const uint b ) { uint2 u2; u2.x = a, u2.y = b; return u2; }
inline uint2 make_uint2( const uint s ) { return make_uint2( s, s ); }
inline uint2 make_uint2( const uint3& a ) { return make_uint2( a.x, a.y ); }
inline uint2 make_uint2( const int2& a ) { return make_uint2( uint( a.x ), uint( a.y ) ); }
inline float3 make_float3( const float& a, const float& b, const float& c ) { float3 f3; f3.x = a, f3.y = b, f3.z = c; return f3; }
inline float3 make_float3( const float& s ) { return make_float3( s, s, s ); }
inline float3 make_float3( const float2& a ) { return make_float3( a.x, a.y, 0.0f ); }
inline float3 make_float3( const float2& a, const float& s ) { return make_float3( a.x, a.y, s ); }
inline float3 make_float3( const float4& a ) { return make_float3( a.x, a.y, a.z ); }
inline float3 make_float3( const int3& a ) { return make_float3( float( a.x ), float( a.y ), float( a.z ) ); }
inline float3 make_float3( const uint3& a ) { return make_float3( float( a.x ), float( a.y ), float( a.z ) ); }
inline int3 make_int3( const int& a, const int& b, const int& c ) { int3 i3; i3.x = a, i3.y = b, i3.z = c; return i3; }
inline int3 make_int3( const int& s ) { return make_int3( s, s, s ); }
inline int3 make_int3( const int2& a ) { return make_int3( a.x, a.y, 0 ); }
inline int3 make_int3( const int2& a, const int& s ) { return make_int3( a.x, a.y, s ); }
inline int3 make_int3( const uint3& a ) { return make_int3( int( a.x ), int( a.y ), int( a.z ) ); }
inline int3 make_int3( const float3& a ) { return make_int3( int( a.x ), int( a.y ), int( a.z ) ); }
inline int3 make_int3( const float4& a ) { return make_int3( int( a.x ), int( a.y ), int( a.z ) ); }
inline uint3 make_uint3( const uint a, uint b, uint c ) { uint3 u3; u3.x = a, u3.y = b, u3.z = c; return u3; }
inline uint3 make_uint3( const uint s ) { return make_uint3( s, s, s ); }
inline uint3 make_uint3( const uint2& a ) { return make_uint3( a.x, a.y, 0 ); }
inline uint3 make_uint3( const uint2& a, const uint s ) { return make_uint3( a.x, a.y, s ); }
inline uint3 make_uint3( const uint4& a ) { return make_uint3( a.x, a.y, a.z ); }
inline uint3 make_uint3( const int3& a ) { return make_uint3( uint( a.x ), uint( a.y ), uint( a.z ) ); }
inline float4 make_float4( const float a, const float b, const float c, const float d ) { float4 f4; f4.x = a, f4.y = b, f4.z = c, f4.w = d; return f4; }
inline float4 make_float4( const float s ) { return make_float4( s, s, s, s ); }
inline float4 make_float4( const float3& a ) { return make_float4( a.x, a.y, a.z, 0.0f ); }
inline float4 make_float4( const float3& a, const float w ) { return make_float4( a.x, a.y, a.z, w ); }
inline float4 make_float4( const int3& a, const float w ) { return make_float4( (float)a.x, (float)a.y, (float)a.z, w ); }
inline float4 make_float4( const int4& a ) { return make_float4( float( a.x ), float( a.y ), float( a.z ), float( a.w ) ); }
inline float4 make_float4( const uint4& a ) { return make_float4( float( a.x ), float( a.y ), float( a.z ), float( a.w ) ); }
inline int4 make_int4( const int a, const int b, const int c, const int d ) { int4 i4; i4.x = a, i4.y = b, i4.z = c, i4.w = d; return i4; }
inline int4 make_int4( const int s ) { return make_int4( s, s, s, s ); }
inline int4 make_int4( const int3& a ) { return make_int4( a.x, a.y, a.z, 0 ); }
inline int4 make_int4( const int3& a, const int w ) { return make_int4( a.x, a.y, a.z, w ); }
inline int4 make_int4( const uint4& a ) { return make_int4( int( a.x ), int( a.y ), int( a.z ), int( a.w ) ); }
inline int4 make_int4( const float4& a ) { return make_int4( int( a.x ), int( a.y ), int( a.z ), int( a.w ) ); }
inline uint4 make_uint4( const uint a, const uint b, const uint c, const uint d ) { uint4 u4; u4.x = a, u4.y = b, u4.z = c, u4.w = d; return u4; }
inline uint4 make_uint4( const uint s ) { return make_uint4( s, s, s, s ); }
inline uint4 make_uint4( const uint3& a ) { return make_uint4( a.x, a.y, a.z, 0 ); }
inline uint4 make_uint4( const uint3& a, const uint w ) { return make_uint4( a.x, a.y, a.z, w ); }
inline uint4 make_uint4( const int4& a ) { return make_uint4( uint( a.x ), uint( a.y ), uint( a.z ), uint( a.w ) ); }
inline uchar4 make_uchar4( const uchar a, const uchar b, const uchar c, const uchar d ) { uchar4 c4; c4.x = a, c4.y = b, c4.z = c, c4.w = d; return c4; }

inline float2 operator-( const float2& a ) { return make_float2( -a.x, -a.y ); }
inline int2 operator-( const int2& a ) { return make_int2( -a.x, -a.y ); }
inline float3 operator-( const float3& a ) { return make_float3( -a.x, -a.y, -a.z ); }
inline int3 operator-( const int3& a ) { return make_int3( -a.x, -a.y, -a.z ); }
inline float4 operator-( const float4& a ) { return make_float4( -a.x, -a.y, -a.z, -a.w ); }
inline int4 operator-( const int4& a ) { return make_int4( -a.x, -a.y, -a.z, -a.w ); }
inline int2 operator << ( const int2& a, int b ) { return make_int2( a.x << b, a.y << b ); }
inline int2 operator >> ( const int2& a, int b ) { return make_int2( a.x >> b, a.y >> b ); }
inline int3 operator << ( const int3& a, int b ) { return make_int3( a.x << b, a.y << b, a.z << b ); }
inline int3 operator >> ( const int3& a, int b ) { return make_int3( a.x >> b, a.y >> b, a.z >> b ); }
inline int4 operator << ( const int4& a, int b ) { return make_int4( a.x << b, a.y << b, a.z << b, a.w << b ); }
inline int4 operator >> ( const int4& a, int b ) { return make_int4( a.x >> b, a.y >> b, a.z >> b, a.w >> b ); }

inline float2 operator+( const float2& a, const float2& b ) { return make_float2( a.x + b.x, a.y + b.y ); }
inline float2 operator+( const float2& a, const int2& b ) { return make_float2( a.x + (float)b.x, a.y + (float)b.y ); }
inline float2 operator+( const float2& a, const uint2& b ) { return make_float2( a.x + (float)b.x, a.y + (float)b.y ); }
inline float2 operator+( const int2& a, const float2& b ) { return make_float2( (float)a.x + b.x, (float)a.y + b.y ); }
inline float2 operator+( const uint2& a, const float2& b ) { return make_float2( (float)a.x + b.x, (float)a.y + b.y ); }
inline void operator+=( float2& a, const float2& b ) { a.x += b.x;	a.y += b.y; }
inline void operator+=( float2& a, const int2& b ) { a.x += (float)b.x; a.y += (float)b.y; }
inline void operator+=( float2& a, const uint2& b ) { a.x += (float)b.x; a.y += (float)b.y; }
inline float2 operator+( const float2& a, float b ) { return make_float2( a.x + b, a.y + b ); }
inline float2 operator+( const float2& a, int b ) { return make_float2( a.x + (float)b, a.y + (float)b ); }
inline float2 operator+( const float2& a, uint b ) { return make_float2( a.x + (float)b, a.y + (float)b ); }
inline float2 operator+( float b, const float2& a ) { return make_float2( a.x + b, a.y + b ); }
inline void operator+=( float2& a, float b ) { a.x += b; a.y += b; }
inline void operator+=( float2& a, int b ) { a.x += (float)b; a.y += (float)b; }
inline void operator+=( float2& a, uint b ) { a.x += (float)b;	a.y += (float)b; }
inline int2 operator+( const int2& a, const int2& b ) { return make_int2( a.x + b.x, a.y + b.y ); }
inline void operator+=( int2& a, const int2& b ) { a.x += b.x;	a.y += b.y; }
inline int2 operator+( const int2& a, int b ) { return make_int2( a.x + b, a.y + b ); }
inline int2 operator+( int b, const int2& a ) { return make_int2( a.x + b, a.y + b ); }
inline void operator+=( int2& a, int b ) { a.x += b;	a.y += b; }
inline uint2 operator+( const uint2& a, const uint2& b ) { return make_uint2( a.x + b.x, a.y + b.y ); }
inline void operator+=( uint2& a, const uint2& b ) { a.x += b.x;	a.y += b.y; }
inline uint2 operator+( const uint2& a, uint b ) { return make_uint2( a.x + b, a.y + b ); }
inline uint2 operator+( uint b, const uint2& a ) { return make_uint2( a.x + b, a.y + b ); }
inline void operator+=( uint2& a, uint b ) { a.x += b;	a.y += b; }
inline float3 operator+( const float3& a, const float3& b ) { return make_float3( a.x + b.x, a.y + b.y, a.z + b.z ); }
inline float3 operator+( const float3& a, const int3& b ) { return make_float3( a.x + (float)b.x, a.y + (float)b.y, a.z + (float)b.z ); }
inline float3 operator+( const float3& a, const uint3& b ) { return make_float3( a.x + (float)b.x, a.y + (float)b.y, a.z + (float)b.z ); }
inline float3 operator+( const int3& a, const float3& b ) { return make_float3( (float)a.x + b.x, (float)a.y + b.y, (float)a.z + b.z ); }
inline float3 operator+( const uint3& a, const float3& b ) { return make_float3( (float)a.x + b.x, (float)a.y + b.y, (float)a.z + b.z ); }
inline void operator+=( float3& a, const float3& b ) { a.x += b.x;	a.y += b.y;	a.z += b.z; }
inline void operator+=( float3& a, const int3& b ) { a.x += (float)b.x; a.y += (float)b.y; a.z += (float)b.z; }
inline void operator+=( float3& a, const uint3& b ) { a.x += (float)b.x; a.y += (float)b.y; a.z += (float)b.z; }
inline float3 operator+( const float3& a, float b ) { return make_float3( a.x + b, a.y + b, a.z + b ); }
inline float3 operator+( const float3& a, int b ) { return make_float3( a.x + (float)b, a.y + (float)b, a.z + (float)b ); }
inline float3 operator+( const float3& a, uint b ) { return make_float3( a.x + (float)b, a.y + (float)b, a.z + (float)b ); }
inline void operator+=( float3& a, float b ) { a.x += b; a.y += b;	a.z += b; }
inline void operator+=( float3& a, int b ) { a.x += (float)b; a.y += (float)b; a.z += (float)b; }
inline void operator+=( float3& a, uint b ) { a.x += (float)b; a.y += (float)b; a.z += (float)b; }
inline int3 operator+( const int3& a, const int3& b ) { return make_int3( a.x + b.x, a.y + b.y, a.z + b.z ); }
inline void operator+=( int3& a, const int3& b ) { a.x += b.x;	a.y += b.y;	a.z += b.z; }
inline int3 operator+( const int3& a, int b ) { return make_int3( a.x + b, a.y + b, a.z + b ); }
inline void operator+=( int3& a, int b ) { a.x += b;	a.y += b;	a.z += b; }
inline uint3 operator+( const uint3& a, const uint3& b ) { return make_uint3( a.x + b.x, a.y + b.y, a.z + b.z ); }
inline void operator+=( uint3& a, const uint3& b ) { a.x += b.x;	a.y += b.y;	a.z += b.z; }
inline uint3 operator+( const uint3& a, uint b ) { return make_uint3( a.x + b, a.y + b, a.z + b ); }
inline void operator+=( uint3& a, uint b ) { a.x += b;	a.y += b;	a.z += b; }
inline int3 operator+( int b, const int3& a ) { return make_int3( a.x + b, a.y + b, a.z + b ); }
inline uint3 operator+( uint b, const uint3& a ) { return make_uint3( a.x + b, a.y + b, a.z + b ); }
inline float3 operator+( float b, const float3& a ) { return make_float3( a.x + b, a.y + b, a.z + b ); }
inline float4 operator+( const float4& a, const float4& b ) { return make_float4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w ); }
inline float4 operator+( const float4& a, const int4& b ) { return make_float4( a.x + (float)b.x, a.y + (float)b.y, a.z + (float)b.z, a.w + (float)b.w ); }
inline float4 operator+( const float4& a, const uint4& b ) { return make_float4( a.x + (float)b.x, a.y + (float)b.y, a.z + (float)b.z, a.w + (float)b.w ); }
inline float4 operator+( const int4& a, const float4& b ) { return make_float4( (float)a.x + b.x, (float)a.y + b.y, (float)a.z + b.z, (float)a.w + b.w ); }
inline float4 operator+( const uint4& a, const float4& b ) { return make_float4( (float)a.x + b.x, (float)a.y + b.y, (float)a.z + b.z, (float)a.w + b.w ); }
inline void operator+=( float4& a, const float4& b ) { a.x += b.x;	a.y += b.y;	a.z += b.z;	a.w += b.w; }
inline void operator+=( float4& a, const int4& b ) { a.x += (float)b.x; a.y += (float)b.y; a.z += (float)b.z; a.w += (float)b.w; }
inline void operator+=( float4& a, const uint4& b ) { a.x += (float)b.x; a.y += (float)b.y; a.z += (float)b.z; a.w += (float)b.w; }
inline float4 operator+( const float4& a, float b ) { return make_float4( a.x + b, a.y + b, a.z + b, a.w + b ); }
inline float4 operator+( const float4& a, int b ) { return make_float4( a.x + (float)b, a.y + (float)b, a.z + (float)b, a.w + (float)b ); }
inline float4 operator+( const float4& a, uint b ) { return make_float4( a.x + (float)b, a.y + (float)b, a.z + (float)b, a.w + (float)b ); }
inline float4 operator+( float b, const float4& a ) { return make_float4( a.x + b, a.y + b, a.z + b, a.w + b ); }
inline void operator+=( float4& a, float b ) { a.x += b;	a.y += b;	a.z += b;	a.w += b; }
inline void operator+=( float4& a, int b ) { a.x += (float)b; a.y += (float)b; a.z += (float)b; a.w += (float)b; }
inline void operator+=( float4& a, uint b ) { a.x += (float)b; a.y += (float)b; a.z += (float)b; a.w += (float)b; }
inline int4 operator+( const int4& a, const int4& b ) { return make_int4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w ); }
inline void operator+=( int4& a, const int4& b ) { a.x += b.x;	a.y += b.y;	a.z += b.z;	a.w += b.w; }
inline int4 operator+( const int4& a, int b ) { return make_int4( a.x + b, a.y + b, a.z + b, a.w + b ); }
inline int4 operator+( int b, const int4& a ) { return make_int4( a.x + b, a.y + b, a.z + b, a.w + b ); }
inline void operator+=( int4& a, int b ) { a.x += b;	a.y += b;	a.z += b;	a.w += b; }
inline uint4 operator+( const uint4& a, const uint4& b ) { return make_uint4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w ); }
inline void operator+=( uint4& a, const uint4& b ) { a.x += b.x;	a.y += b.y;	a.z += b.z;	a.w += b.w; }
inline uint4 operator+( const uint4& a, uint b ) { return make_uint4( a.x + b, a.y + b, a.z + b, a.w + b ); }
inline uint4 operator+( uint b, const uint4& a ) { return make_uint4( a.x + b, a.y + b, a.z + b, a.w + b ); }
inline void operator+=( uint4& a, uint b ) { a.x += b;	a.y += b;	a.z += b;	a.w += b; }

inline float2 operator-( const float2& a, const float2& b ) { return make_float2( a.x - b.x, a.y - b.y ); }
inline float2 operator-( const float2& a, const int2& b ) { return make_float2( a.x - (float)b.x, a.y - (float)b.y ); }
inline float2 operator-( const float2& a, const uint2& b ) { return make_float2( a.x - (float)b.x, a.y - (float)b.y ); }
inline float2 operator-( const int2& a, const float2& b ) { return make_float2( (float)a.x - b.x, (float)a.y - b.y ); }
inline float2 operator-( const uint2& a, const float2& b ) { return make_float2( (float)a.x - b.x, (float)a.y - b.y ); }
inline void operator-=( float2& a, const float2& b ) { a.x -= b.x;	a.y -= b.y; }
inline void operator-=( float2& a, const int2& b ) { a.x -= (float)b.x; a.y -= (float)b.y; }
inline void operator-=( float2& a, const uint2& b ) { a.x -= (float)b.x; a.y -= (float)b.y; }
inline float2 operator-( const float2& a, float b ) { return make_float2( a.x - b, a.y - b ); }
inline float2 operator-( const float2& a, int b ) { return make_float2( a.x - (float)b, a.y - (float)b ); }
inline float2 operator-( const float2& a, uint b ) { return make_float2( a.x - (float)b, a.y - (float)b ); }
inline float2 operator-( float b, const float2& a ) { return make_float2( b - a.x, b - a.y ); }
inline void operator-=( float2& a, float b ) { a.x -= b; a.y -= b; }
inline void operator-=( float2& a, int b ) { a.x -= (float)b; a.y -= (float)b; }
inline void operator-=( float2& a, uint b ) { a.x -= (float)b; a.y -= (float)b; }
inline int2 operator-( const int2& a, const int2& b ) { return make_int2( a.x - b.x, a.y - b.y ); }
inline void operator-=( int2& a, const int2& b ) { a.x -= b.x;	a.y -= b.y; }
inline int2 operator-( const int2& a, int b ) { return make_int2( a.x - b, a.y - b ); }
inline int2 operator-( int b, const int2& a ) { return make_int2( b - a.x, b - a.y ); }
inline void operator-=( int2& a, int b ) { a.x -= b;	a.y -= b; }
inline uint2 operator-( const uint2& a, const uint2& b ) { return make_uint2( a.x - b.x, a.y - b.y ); }
inline void operator-=( uint2& a, const uint2& b ) { a.x -= b.x;	a.y -= b.y; }
inline uint2 operator-( const uint2& a, uint b ) { return make_uint2( a.x - b, a.y - b ); }
inline uint2 operator-( uint b, const uint2& a ) { return make_uint2( b - a.x, b - a.y ); }
inline void operator-=( uint2& a, uint b ) { a.x -= b;	a.y -= b; }
inline float3 operator-( const float3& a, const float3& b ) { return make_float3( a.x - b.x, a.y - b.y, a.z - b.z ); }
inline float3 operator-( const float3& a, const int3& b ) { return make_float3( a.x - (float)b.x, a.y - (float)b.y, a.z - (float)b.z ); }
inline float3 operator-( const float3& a, const uint3& b ) { return make_float3( a.x - (float)b.x, a.y - (float)b.y, a.z - (float)b.z ); }
inline float3 operator-( const int3& a, const float3& b ) { return make_float3( (float)a.x - b.x, (float)a.y - b.y, (float)a.z - b.z ); }
inline float3 operator-( const uint3& a, const float3& b ) { return make_float3( (float)a.x - b.x, (float)a.y - b.y, (float)a.z - b.z ); }
inline void operator-=( float3& a, const float3& b ) { a.x -= b.x;	a.y -= b.y;	a.z -= b.z; }
inline void operator-=( float3& a, const int3& b ) { a.x -= (float)b.x; a.y -= (float)b.y; a.z -= (float)b.z; }
inline void operator-=( float3& a, const uint3& b ) { a.x -= (float)b.x; a.y -= (float)b.y; a.z -= (float)b.z; }
inline float3 operator-( const float3& a, float b ) { return make_float3( a.x - b, a.y - b, a.z - b ); }
inline float3 operator-( const float3& a, int b ) { return make_float3( a.x - (float)b, a.y - (float)b, a.z - (float)b ); }
inline float3 operator-( const float3& a, uint b ) { return make_float3( a.x - (float)b, a.y - (float)b, a.z - (float)b ); }
inline float3 operator-( float b, const float3& a ) { return make_float3( b - a.x, b - a.y, b - a.z ); }
inline void operator-=( float3& a, float b ) { a.x -= b; a.y -= b; a.z -= b; }
inline void operator-=( float3& a, int b ) { a.x -= (float)b; a.y -= (float)b; a.z -= (float)b; }
inline void operator-=( float3& a, uint b ) { a.x -= (float)b;	a.y -= (float)b; a.z -= (float)b; }
inline int3 operator-( const int3& a, const int3& b ) { return make_int3( a.x - b.x, a.y - b.y, a.z - b.z ); }
inline void operator-=( int3& a, const int3& b ) { a.x -= b.x;	a.y -= b.y;	a.z -= b.z; }
inline int3 operator-( const int3& a, int b ) { return make_int3( a.x - b, a.y - b, a.z - b ); }
inline int3 operator-( int b, const int3& a ) { return make_int3( b - a.x, b - a.y, b - a.z ); }
inline void operator-=( int3& a, int b ) { a.x -= b;	a.y -= b;	a.z -= b; }
inline uint3 operator-( const uint3& a, const uint3& b ) { return make_uint3( a.x - b.x, a.y - b.y, a.z - b.z ); }
inline void operator-=( uint3& a, const uint3& b ) { a.x -= b.x;	a.y -= b.y;	a.z -= b.z; }
inline uint3 operator-( const uint3& a, uint b ) { return make_uint3( a.x - b, a.y - b, a.z - b ); }
inline uint3 operator-( uint b, const uint3& a ) { return make_uint3( b - a.x, b - a.y, b - a.z ); }
inline void operator-=( uint3& a, uint b ) { a.x -= b;	a.y -= b;	a.z -= b; }
inline float4 operator-( const float4& a, const float4& b ) { return make_float4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w ); }
inline float4 operator-( const float4& a, const int4& b ) { return make_float4( a.x - (float)b.x, a.y - (float)b.y, a.z - (float)b.z, a.w - (float)b.w ); }
inline float4 operator-( const float4& a, const uint4& b ) { return make_float4( a.x - (float)b.x, a.y - (float)b.y, a.z - (float)b.z, a.w - (float)b.w ); }
inline float4 operator-( const int4& a, const float4& b ) { return make_float4( (float)a.x - b.x, (float)a.y - b.y, (float)a.z - b.z, (float)a.w - b.w ); }
inline float4 operator-( const uint4& a, const float4& b ) { return make_float4( (float)a.x - b.x, (float)a.y - b.y, (float)a.z - b.z, (float)a.w - b.w ); }
inline void operator-=( float4& a, const float4& b ) { a.x -= b.x;	a.y -= b.y;	a.z -= b.z;	a.w -= b.w; }
inline void operator-=( float4& a, const int4& b ) { a.x -= (float)b.x; a.y -= (float)b.y; a.z -= (float)b.z; a.w -= (float)b.w; }
inline void operator-=( float4& a, const uint4& b ) { a.x -= (float)b.x; a.y -= (float)b.y; a.z -= (float)b.z; a.w -= (float)b.w; }
inline float4 operator-( const float4& a, float b ) { return make_float4( a.x - b, a.y - b, a.z - b, a.w - b ); }
inline float4 operator-( const float4& a, int b ) { return make_float4( a.x - (float)b, a.y - (float)b, a.z - (float)b, a.w - (float)b ); }
inline float4 operator-( const float4& a, uint b ) { return make_float4( a.x - (float)b, a.y - (float)b, a.z - (float)b, a.w - (float)b ); }
inline void operator-=( float4& a, float b ) { a.x -= b; a.y -= b; a.z -= b; a.w -= b; }
inline void operator-=( float4& a, int b ) { a.x -= (float)b; a.y -= (float)b; a.z -= (float)b; a.w -= (float)b; }
inline void operator-=( float4& a, uint b ) { a.x -= (float)b; a.y -= (float)b; a.z -= (float)b; a.w -= (float)b; }
inline int4 operator-( const int4& a, const int4& b ) { return make_int4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w ); }
inline void operator-=( int4& a, const int4& b ) { a.x -= b.x;	a.y -= b.y;	a.z -= b.z;	a.w -= b.w; }
inline int4 operator-( const int4& a, int b ) { return make_int4( a.x - b, a.y - b, a.z - b, a.w - b ); }
inline int4 operator-( int b, const int4& a ) { return make_int4( b - a.x, b - a.y, b - a.z, b - a.w ); }
inline void operator-=( int4& a, int b ) { a.x -= b;	a.y -= b;	a.z -= b;	a.w -= b; }
inline uint4 operator-( const uint4& a, const uint4& b ) { return make_uint4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w ); }
inline void operator-=( uint4& a, const uint4& b ) { a.x -= b.x;	a.y -= b.y;	a.z -= b.z;	a.w -= b.w; }
inline uint4 operator-( const uint4& a, uint b ) { return make_uint4( a.x - b, a.y - b, a.z - b, a.w - b ); }
inline uint4 operator-( uint b, const uint4& a ) { return make_uint4( b - a.x, b - a.y, b - a.z, b - a.w ); }
inline void operator-=( uint4& a, uint b ) { a.x -= b;	a.y -= b;	a.z -= b;	a.w -= b; }

inline float2 operator*( const float2& a, const float2& b ) { return make_float2( a.x * b.x, a.y * b.y ); }
inline void operator*=( float2& a, const float2& b ) { a.x *= b.x;	a.y *= b.y; }
inline float2 operator*( const float2& a, float b ) { return make_float2( a.x * b, a.y * b ); }
inline float2 operator*( float b, const float2& a ) { return make_float2( b * a.x, b * a.y ); }
inline void operator*=( float2& a, float b ) { a.x *= b;	a.y *= b; }
inline int2 operator*( const int2& a, const int2& b ) { return make_int2( a.x * b.x, a.y * b.y ); }
inline void operator*=( int2& a, const int2& b ) { a.x *= b.x;	a.y *= b.y; }
inline int2 operator*( const int2& a, int b ) { return make_int2( a.x * b, a.y * b ); }
inline int2 operator*( int b, const int2& a ) { return make_int2( b * a.x, b * a.y ); }
inline void operator*=( int2& a, int b ) { a.x *= b;	a.y *= b; }
inline uint2 operator*( const uint2& a, const uint2& b ) { return make_uint2( a.x * b.x, a.y * b.y ); }
inline void operator*=( uint2& a, const uint2& b ) { a.x *= b.x;	a.y *= b.y; }
inline uint2 operator*( const uint2& a, uint b ) { return make_uint2( a.x * b, a.y * b ); }
inline uint2 operator*( uint b, const uint2& a ) { return make_uint2( b * a.x, b * a.y ); }
inline void operator*=( uint2& a, uint b ) { a.x *= b;	a.y *= b; }
inline float3 operator*( const float3& a, const float3& b ) { return make_float3( a.x * b.x, a.y * b.y, a.z * b.z ); }
inline void operator*=( float3& a, const float3& b ) { a.x *= b.x;	a.y *= b.y;	a.z *= b.z; }
inline float3 operator*( const float3& a, float b ) { return make_float3( a.x * b, a.y * b, a.z * b ); }
inline float3 operator*( float b, const float3& a ) { return make_float3( b * a.x, b * a.y, b * a.z ); }
inline void operator*=( float3& a, float b ) { a.x *= b;	a.y *= b;	a.z *= b; }
inline int3 operator*( const int3& a, const int3& b ) { return make_int3( a.x * b.x, a.y * b.y, a.z * b.z ); }
inline void operator*=( int3& a, const int3& b ) { a.x *= b.x;	a.y *= b.y;	a.z *= b.z; }
inline int3 operator*( const int3& a, int b ) { return make_int3( a.x * b, a.y * b, a.z * b ); }
inline int3 operator*( int b, const int3& a ) { return make_int3( b * a.x, b * a.y, b * a.z ); }
inline void operator*=( int3& a, int b ) { a.x *= b;	a.y *= b;	a.z *= b; }
inline uint3 operator*( const uint3& a, const uint3& b ) { return make_uint3( a.x * b.x, a.y * b.y, a.z * b.z ); }
inline void operator*=( uint3& a, const uint3& b ) { a.x *= b.x;	a.y *= b.y;	a.z *= b.z; }
inline uint3 operator*( const uint3& a, uint b ) { return make_uint3( a.x * b, a.y * b, a.z * b ); }
inline uint3 operator*( uint b, const uint3& a ) { return make_uint3( b * a.x, b * a.y, b * a.z ); }
inline void operator*=( uint3& a, uint b ) { a.x *= b;	a.y *= b;	a.z *= b; }
inline float4 operator*( const float4& a, const float4& b ) { return make_float4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w ); }
inline void operator*=( float4& a, const float4& b ) { a.x *= b.x;	a.y *= b.y;	a.z *= b.z;	a.w *= b.w; }
inline float4 operator*( const float4& a, float b ) { return make_float4( a.x * b, a.y * b, a.z * b, a.w * b ); }
inline float4 operator*( float b, const float4& a ) { return make_float4( b * a.x, b * a.y, b * a.z, b * a.w ); }
inline void operator*=( float4& a, float b ) { a.x *= b;	a.y *= b;	a.z *= b;	a.w *= b; }
inline int4 operator*( const int4& a, const int4& b ) { return make_int4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w ); }
inline void operator*=( int4& a, const int4& b ) { a.x *= b.x;	a.y *= b.y;	a.z *= b.z;	a.w *= b.w; }
inline int4 operator*( const int4& a, int b ) { return make_int4( a.x * b, a.y * b, a.z * b, a.w * b ); }
inline int4 operator*( int b, const int4& a ) { return make_int4( b * a.x, b * a.y, b * a.z, b * a.w ); }
inline void operator*=( int4& a, int b ) { a.x *= b;	a.y *= b;	a.z *= b;	a.w *= b; }
inline uint4 operator*( const uint4& a, const uint4& b ) { return make_uint4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w ); }
inline void operator*=( uint4& a, const uint4& b ) { a.x *= b.x;	a.y *= b.y;	a.z *= b.z;	a.w *= b.w; }
inline uint4 operator*( const uint4& a, uint b ) { return make_uint4( a.x * b, a.y * b, a.z * b, a.w * b ); }
inline uint4 operator*( uint b, const uint4& a ) { return make_uint4( b * a.x, b * a.y, b * a.z, b * a.w ); }
inline void operator*=( uint4& a, uint b ) { a.x *= b;	a.y *= b;	a.z *= b;	a.w *= b; }

inline float2 operator/( const float2& a, const float2& b ) { return make_float2( a.x / b.x, a.y / b.y ); }
inline void operator/=( float2& a, const float2& b ) { a.x /= b.x;	a.y /= b.y; }
inline float2 operator/( const float2& a, float b ) { return make_float2( a.x / b, a.y / b ); }
inline void operator/=( float2& a, float b ) { a.x /= b;	a.y /= b; }
inline float2 operator/( float b, const float2& a ) { return make_float2( b / a.x, b / a.y ); }
inline float3 operator/( const float3& a, const float3& b ) { return make_float3( a.x / b.x, a.y / b.y, a.z / b.z ); }
inline void operator/=( float3& a, const float3& b ) { a.x /= b.x;	a.y /= b.y;	a.z /= b.z; }
inline float3 operator/( const float3& a, float b ) { return make_float3( a.x / b, a.y / b, a.z / b ); }
inline void operator/=( float3& a, float b ) { a.x /= b;	a.y /= b;	a.z /= b; }
inline float3 operator/( float b, const float3& a ) { return make_float3( b / a.x, b / a.y, b / a.z ); }
inline float4 operator/( const float4& a, const float4& b ) { return make_float4( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w ); }
inline void operator/=( float4& a, const float4& b ) { a.x /= b.x;	a.y /= b.y;	a.z /= b.z;	a.w /= b.w; }
inline float4 operator/( const float4& a, float b ) { return make_float4( a.x / b, a.y / b, a.z / b, a.w / b ); }
inline void operator/=( float4& a, float b ) { a.x /= b;	a.y /= b;	a.z /= b;	a.w /= b; }
inline float4 operator/( float b, const float4& a ) { return make_float4( b / a.x, b / a.y, b / a.z, b / a.w ); }

inline float2 fminf( const float2& a, const float2& b ) { return make_float2( fminf( a.x, b.x ), fminf( a.y, b.y ) ); }
inline float3 fminf( const float3& a, const float3& b ) { return make_float3( fminf( a.x, b.x ), fminf( a.y, b.y ), fminf( a.z, b.z ) ); }
inline float4 fminf( const float4& a, const float4& b ) { return make_float4( fminf( a.x, b.x ), fminf( a.y, b.y ), fminf( a.z, b.z ), fminf( a.w, b.w ) ); }
inline int2 min( const int2& a, const int2& b ) { return make_int2( min( a.x, b.x ), min( a.y, b.y ) ); }
inline int3 min( const int3& a, const int3& b ) { return make_int3( min( a.x, b.x ), min( a.y, b.y ), min( a.z, b.z ) ); }
inline int4 min( const int4& a, const int4& b ) { return make_int4( min( a.x, b.x ), min( a.y, b.y ), min( a.z, b.z ), min( a.w, b.w ) ); }
inline uint2 min( const uint2& a, const uint2& b ) { return make_uint2( min( a.x, b.x ), min( a.y, b.y ) ); }
inline uint3 min( const uint3& a, const uint3& b ) { return make_uint3( min( a.x, b.x ), min( a.y, b.y ), min( a.z, b.z ) ); }
inline uint4 min( const uint4& a, const uint4& b ) { return make_uint4( min( a.x, b.x ), min( a.y, b.y ), min( a.z, b.z ), min( a.w, b.w ) ); }

inline float2 fmaxf( const float2& a, const float2& b ) { return make_float2( fmaxf( a.x, b.x ), fmaxf( a.y, b.y ) ); }
inline float3 fmaxf( const float3& a, const float3& b ) { return make_float3( fmaxf( a.x, b.x ), fmaxf( a.y, b.y ), fmaxf( a.z, b.z ) ); }
inline float4 fmaxf( const float4& a, const float4& b ) { return make_float4( fmaxf( a.x, b.x ), fmaxf( a.y, b.y ), fmaxf( a.z, b.z ), fmaxf( a.w, b.w ) ); }
inline int2 max( const int2& a, const int2& b ) { return make_int2( max( a.x, b.x ), max( a.y, b.y ) ); }
inline int3 max( const int3& a, const int3& b ) { return make_int3( max( a.x, b.x ), max( a.y, b.y ), max( a.z, b.z ) ); }
inline int4 max( const int4& a, const int4& b ) { return make_int4( max( a.x, b.x ), max( a.y, b.y ), max( a.z, b.z ), max( a.w, b.w ) ); }
inline uint2 max( const uint2& a, const uint2& b ) { return make_uint2( max( a.x, b.x ), max( a.y, b.y ) ); }
inline uint3 max( const uint3& a, const uint3& b ) { return make_uint3( max( a.x, b.x ), max( a.y, b.y ), max( a.z, b.z ) ); }
inline uint4 max( const uint4& a, const uint4& b ) { return make_uint4( max( a.x, b.x ), max( a.y, b.y ), max( a.z, b.z ), max( a.w, b.w ) ); }

inline float lerp( float a, float b, float t ) { return a + t * (b - a); }
inline float2 lerp( const float2& a, const float2& b, float t ) { return a + t * (b - a); }
inline float3 lerp( const float3& a, const float3& b, float t ) { return a + t * (b - a); }
inline float4 lerp( const float4& a, const float4& b, float t ) { return a + t * (b - a); }

inline float clamp( float f, float a, float b ) { return fmaxf( a, fminf( f, b ) ); }
inline int clamp( int f, int a, int b ) { return max( a, min( f, b ) ); }
inline uint clamp( uint f, uint a, uint b ) { return max( a, min( f, b ) ); }
inline float2 clamp( const float2& v, float a, float b ) { return make_float2( clamp( v.x, a, b ), clamp( v.y, a, b ) ); }
inline float2 clamp( const float2& v, const float2& a, const float2& b ) { return make_float2( clamp( v.x, a.x, b.x ), clamp( v.y, a.y, b.y ) ); }
inline float3 clamp( const float3& v, float a, float b ) { return make_float3( clamp( v.x, a, b ), clamp( v.y, a, b ), clamp( v.z, a, b ) ); }
inline float3 clamp( const float3& v, const float3& a, const float3& b ) { return make_float3( clamp( v.x, a.x, b.x ), clamp( v.y, a.y, b.y ), clamp( v.z, a.z, b.z ) ); }
inline float4 clamp( const float4& v, float a, float b ) { return make_float4( clamp( v.x, a, b ), clamp( v.y, a, b ), clamp( v.z, a, b ), clamp( v.w, a, b ) ); }
inline float4 clamp( const float4& v, const float4& a, const float4& b ) { return make_float4( clamp( v.x, a.x, b.x ), clamp( v.y, a.y, b.y ), clamp( v.z, a.z, b.z ), clamp( v.w, a.w, b.w ) ); }
inline int2 clamp( const int2& v, int a, int b ) { return make_int2( clamp( v.x, a, b ), clamp( v.y, a, b ) ); }
inline int2 clamp( const int2& v, const int2& a, const int2& b ) { return make_int2( clamp( v.x, a.x, b.x ), clamp( v.y, a.y, b.y ) ); }
inline int3 clamp( const int3& v, int a, int b ) { return make_int3( clamp( v.x, a, b ), clamp( v.y, a, b ), clamp( v.z, a, b ) ); }
inline int3 clamp( const int3& v, const int3& a, const int3& b ) { return make_int3( clamp( v.x, a.x, b.x ), clamp( v.y, a.y, b.y ), clamp( v.z, a.z, b.z ) ); }
inline int4 clamp( const int4& v, int a, int b ) { return make_int4( clamp( v.x, a, b ), clamp( v.y, a, b ), clamp( v.z, a, b ), clamp( v.w, a, b ) ); }
inline int4 clamp( const int4& v, const int4& a, const int4& b ) { return make_int4( clamp( v.x, a.x, b.x ), clamp( v.y, a.y, b.y ), clamp( v.z, a.z, b.z ), clamp( v.w, a.w, b.w ) ); }
inline uint2 clamp( const uint2& v, uint a, uint b ) { return make_uint2( clamp( v.x, a, b ), clamp( v.y, a, b ) ); }
inline uint2 clamp( const uint2& v, const uint2& a, const uint2& b ) { return make_uint2( clamp( v.x, a.x, b.x ), clamp( v.y, a.y, b.y ) ); }
inline uint3 clamp( const uint3& v, uint a, uint b ) { return make_uint3( clamp( v.x, a, b ), clamp( v.y, a, b ), clamp( v.z, a, b ) ); }
inline uint3 clamp( const uint3& v, const uint3& a, const uint3& b ) { return make_uint3( clamp( v.x, a.x, b.x ), clamp( v.y, a.y, b.y ), clamp( v.z, a.z, b.z ) ); }
inline uint4 clamp( const uint4& v, uint a, uint b ) { return make_uint4( clamp( v.x, a, b ), clamp( v.y, a, b ), clamp( v.z, a, b ), clamp( v.w, a, b ) ); }
inline uint4 clamp( const uint4& v, const uint4& a, const uint4& b ) { return make_uint4( clamp( v.x, a.x, b.x ), clamp( v.y, a.y, b.y ), clamp( v.z, a.z, b.z ), clamp( v.w, a.w, b.w ) ); }

inline float dot( const float2& a, const float2& b ) { return a.x * b.x + a.y * b.y; }
inline float dot( const float3& a, const float3& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float dot( const float4& a, const float4& b ) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
inline int dot( const int2& a, const int2& b ) { return a.x * b.x + a.y * b.y; }
inline int dot( const int3& a, const int3& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline int dot( const int4& a, const int4& b ) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
inline uint dot( const uint2& a, const uint2& b ) { return a.x * b.x + a.y * b.y; }
inline uint dot( const uint3& a, const uint3& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline uint dot( const uint4& a, const uint4& b ) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

inline float sqrLength( const float2& v ) { return dot( v, v ); }
inline float sqrLength( const float3& v ) { return dot( v, v ); }
inline float sqrLength( const float4& v ) { return dot( v, v ); }

inline float length( const float2& v ) { return sqrtf( dot( v, v ) ); }
inline float length( const float3& v ) { return sqrtf( dot( v, v ) ); }
inline float length( const float4& v ) { return sqrtf( dot( v, v ) ); }

inline float length( const int2& v ) { return sqrtf( (float)dot( v, v ) ); }
inline float length( const int3& v ) { return sqrtf( (float)dot( v, v ) ); }
inline float length( const int4& v ) { return sqrtf( (float)dot( v, v ) ); }

inline float2 normalize( const float2& v ) { float invLen = rsqrtf( dot( v, v ) ); return v * invLen; }
inline float3 normalize( const float3& v ) { float invLen = rsqrtf( dot( v, v ) ); return v * invLen; }
inline float4 normalize( const float4& v ) { float invLen = rsqrtf( dot( v, v ) ); return v * invLen; }

inline uint dominantAxis( const float2& v ) { float x = fabs( v.x ), y = fabs( v.y ); return x > y ? 0 : 1; } // for coherent grid traversal
inline uint dominantAxis( const float3& v ) { float x = fabs( v.x ), y = fabs( v.y ), z = fabs( v.z ); float m = max( max( x, y ), z ); return m == x ? 0 : (m == y ? 1 : 2); }

inline float2 floorf( const float2& v ) { return make_float2( floorf( v.x ), floorf( v.y ) ); }
inline float3 floorf( const float3& v ) { return make_float3( floorf( v.x ), floorf( v.y ), floorf( v.z ) ); }
inline float4 floorf( const float4& v ) { return make_float4( floorf( v.x ), floorf( v.y ), floorf( v.z ), floorf( v.w ) ); }

inline float fracf( float v ) { return v - floorf( v ); }
inline float2 fracf( const float2& v ) { return make_float2( fracf( v.x ), fracf( v.y ) ); }
inline float3 fracf( const float3& v ) { return make_float3( fracf( v.x ), fracf( v.y ), fracf( v.z ) ); }
inline float4 fracf( const float4& v ) { return make_float4( fracf( v.x ), fracf( v.y ), fracf( v.z ), fracf( v.w ) ); }

inline float2 fmodf( const float2& a, const float2& b ) { return make_float2( fmodf( a.x, b.x ), fmodf( a.y, b.y ) ); }
inline float3 fmodf( const float3& a, const float3& b ) { return make_float3( fmodf( a.x, b.x ), fmodf( a.y, b.y ), fmodf( a.z, b.z ) ); }
inline float4 fmodf( const float4& a, const float4& b ) { return make_float4( fmodf( a.x, b.x ), fmodf( a.y, b.y ), fmodf( a.z, b.z ), fmodf( a.w, b.w ) ); }

inline float2 fabs( const float2& v ) { return make_float2( fabs( v.x ), fabs( v.y ) ); }
inline float3 fabs( const float3& v ) { return make_float3( fabs( v.x ), fabs( v.y ), fabs( v.z ) ); }
inline float4 fabs( const float4& v ) { return make_float4( fabs( v.x ), fabs( v.y ), fabs( v.z ), fabs( v.w ) ); }
inline int2 abs( const int2& v ) { return make_int2( abs( v.x ), abs( v.y ) ); }
inline int3 abs( const int3& v ) { return make_int3( abs( v.x ), abs( v.y ), abs( v.z ) ); }
inline int4 abs( const int4& v ) { return make_int4( abs( v.x ), abs( v.y ), abs( v.z ), abs( v.w ) ); }

inline float3 reflect( const float3& i, const float3& n ) { return i - 2.0f * n * dot( n, i ); }

inline float3 cross( const float3& a, const float3& b ) { return make_float3( a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x ); }

inline float smoothstep( float a, float b, float x )
{
	float y = clamp( (x - a) / (b - a), 0.0f, 1.0f );
	return (y * y * (3.0f - (2.0f * y)));
}
inline float2 smoothstep( float2 a, float2 b, float2 x )
{
	float2 y = clamp( (x - a) / (b - a), 0.0f, 1.0f );
	return (y * y * (make_float2( 3.0f ) - (make_float2( 2.0f ) * y)));
}

inline float3 smoothstep( float3 a, float3 b, float3 x )
{
	float3 y = clamp( (x - a) / (b - a), 0.0f, 1.0f );
	return (y * y * (make_float3( 3.0f ) - (make_float3( 2.0f ) * y)));
}
inline float4 smoothstep( float4 a, float4 b, float4 x )
{
	float4 y = clamp( (x - a) / (b - a), 0.0f, 1.0f );
	return (y * y * (make_float4( 3.0f ) - (make_float4( 2.0f ) * y)));
}

// global project settigs; shared with OpenCL
#include "common.h"

// Add your headers here; they will be able to use all previously defined classes and namespaces.
// In your own .cpp files just add #include "precomp.h".
// #include "my_include.h"


// InstructionSet.cpp
// Compile by using: cl /EHsc /W4 InstructionSet.cpp
// processor: x86, x64
// Uses the __cpuid intrinsic to get information about
// CPU extended instruction set support.

#include <iostream>
#include <bitset>
#include <array>
// #include <intrin.h>

// instruction set detection
#ifdef _WIN32
#define cpuid(info, x) __cpuidex(info, x, 0)
#else
#include <cpuid.h>
void cpuid( int info[4], int InfoType ) { __cpuid_count( InfoType, 0, info[0], info[1], info[2], info[3] ); }
#endif
class CPUCaps // from https://github.com/Mysticial/FeatureDetector
{
public:
	static inline bool HW_MMX = false;
	static inline bool HW_x64 = false;
	static inline bool HW_ABM = false;
	static inline bool HW_RDRAND = false;
	static inline bool HW_BMI1 = false;
	static inline bool HW_BMI2 = false;
	static inline bool HW_ADX = false;
	static inline bool HW_PREFETCHWT1 = false;
	// SIMD: 128-bit
	static inline bool HW_SSE = false;
	static inline bool HW_SSE2 = false;
	static inline bool HW_SSE3 = false;
	static inline bool HW_SSSE3 = false;
	static inline bool HW_SSE41 = false;
	static inline bool HW_SSE42 = false;
	static inline bool HW_SSE4a = false;
	static inline bool HW_AES = false;
	static inline bool HW_SHA = false;
	// SIMD: 256-bit
	static inline bool HW_AVX = false;
	static inline bool HW_XOP = false;
	static inline bool HW_FMA3 = false;
	static inline bool HW_FMA4 = false;
	static inline bool HW_AVX2 = false;
	// SIMD: 512-bit
	static inline bool HW_AVX512F = false;    //  AVX512 Foundation
	static inline bool HW_AVX512CD = false;   //  AVX512 Conflict Detection
	static inline bool HW_AVX512PF = false;   //  AVX512 Prefetch
	static inline bool HW_AVX512ER = false;   //  AVX512 Exponential + Reciprocal
	static inline bool HW_AVX512VL = false;   //  AVX512 Vector Length Extensions
	static inline bool HW_AVX512BW = false;   //  AVX512 Byte + Word
	static inline bool HW_AVX512DQ = false;   //  AVX512 Doubleword + Quadword
	static inline bool HW_AVX512IFMA = false; //  AVX512 Integer 52-bit Fused Multiply-Add
	static inline bool HW_AVX512VBMI = false; //  AVX512 Vector Byte Manipulation Instructions
	// constructor
	CPUCaps()
	{
		int info[4];
		cpuid( info, 0 );
		int nIds = info[0];
		cpuid( info, 0x80000000 );
		unsigned nExIds = info[0];
		// detect Features
		if (nIds >= 0x00000001)
		{
			cpuid( info, 0x00000001 );
			HW_MMX = (info[3] & ((int)1 << 23)) != 0;
			HW_SSE = (info[3] & ((int)1 << 25)) != 0;
			HW_SSE2 = (info[3] & ((int)1 << 26)) != 0;
			HW_SSE3 = (info[2] & ((int)1 << 0)) != 0;
			HW_SSSE3 = (info[2] & ((int)1 << 9)) != 0;
			HW_SSE41 = (info[2] & ((int)1 << 19)) != 0;
			HW_SSE42 = (info[2] & ((int)1 << 20)) != 0;
			HW_AES = (info[2] & ((int)1 << 25)) != 0;
			HW_AVX = (info[2] & ((int)1 << 28)) != 0;
			HW_FMA3 = (info[2] & ((int)1 << 12)) != 0;
			HW_RDRAND = (info[2] & ((int)1 << 30)) != 0;
		}
		if (nIds >= 0x00000007)
		{
			cpuid( info, 0x00000007 );
			HW_AVX2 = (info[1] & ((int)1 << 5)) != 0;
			HW_BMI1 = (info[1] & ((int)1 << 3)) != 0;
			HW_BMI2 = (info[1] & ((int)1 << 8)) != 0;
			HW_ADX = (info[1] & ((int)1 << 19)) != 0;
			HW_SHA = (info[1] & ((int)1 << 29)) != 0;
			HW_PREFETCHWT1 = (info[2] & ((int)1 << 0)) != 0;
			HW_AVX512F = (info[1] & ((int)1 << 16)) != 0;
			HW_AVX512CD = (info[1] & ((int)1 << 28)) != 0;
			HW_AVX512PF = (info[1] & ((int)1 << 26)) != 0;
			HW_AVX512ER = (info[1] & ((int)1 << 27)) != 0;
			HW_AVX512VL = (info[1] & ((int)1 << 31)) != 0;
			HW_AVX512BW = (info[1] & ((int)1 << 30)) != 0;
			HW_AVX512DQ = (info[1] & ((int)1 << 17)) != 0;
			HW_AVX512IFMA = (info[1] & ((int)1 << 21)) != 0;
			HW_AVX512VBMI = (info[2] & ((int)1 << 1)) != 0;
		}
		if (nExIds >= 0x80000001)
		{
			cpuid( info, 0x80000001 );
			HW_x64 = (info[3] & ((int)1 << 29)) != 0;
			HW_ABM = (info[2] & ((int)1 << 5)) != 0;
			HW_SSE4a = (info[2] & ((int)1 << 6)) != 0;
			HW_FMA4 = (info[2] & ((int)1 << 16)) != 0;
			HW_XOP = (info[2] & ((int)1 << 11)) != 0;
		}
	}
};

// application base class
class TheApp
{
public:
	virtual void Init() = 0;
	virtual void Tick( float deltaTime ) = 0;
	// virtual void Shutdown() = 0;
	// virtual void MouseUp( int button ) = 0;
	// virtual void MouseDown( int button ) = 0;
	// virtual void MouseMove( int x, int y ) = 0;
	// virtual void MouseWheel( float y ) = 0;
	// virtual void KeyUp( int key ) = 0;
	// virtual void KeyDown( int key ) = 0;
	// Surface* screen = 0;
};










