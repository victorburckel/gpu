#include "NBodyCPU.h"
#include "ParticleCpu.h"
#include <ppl.h>

using namespace concurrency;
using namespace concurrency::graphics;

namespace {
	const float softeningSquared = 0.0000015625f;
	const float dampingFactor = 0.9995f;
	const float particleMass = ( ( 6.67300e-11f * 10000.0f ) * 10000.0f * 10000.0f );

	inline const float sqrLength( const float_3& r )
	{
		return r.x * r.x + r.y * r.y + r.z * r.z;
	}

	void bodyBodyInteraction( const ParticleCpu* const particlesIn, ParticleCpu& particleOut, int numParticles, float deltaTime )
	{
		float_3 pos( particleOut.pos );
		float_3 vel( particleOut.vel );
		float_3 acc( 0.0f );

		std::for_each( particlesIn, particlesIn + numParticles, [ =, &acc ]( const ParticleCpu& p )
		{
			const float_3 r = p.pos - pos;

			float distSqr = sqrLength( r ) + softeningSquared;
			float invDist = 1.0f / sqrt( distSqr );
			float invDistCube = invDist * invDist * invDist;
			float s = particleMass * invDistCube;

			acc += r * s;
		} );

		vel += acc * deltaTime;
		vel *= dampingFactor;
		pos += vel * deltaTime;

		particleOut.pos = pos;
		particleOut.vel = vel;
	}

	void bodyBodyInteractionSSE( const ParticleCpu* const pParticlesIn, ParticleCpu& particleOut, int numParticles, float deltaTime )
	{
		const __m128 __softeningSquared = _mm_load1_ps( &softeningSquared );
		const __m128 __dampingFactor = _mm_load_ps1( &dampingFactor );
		const __m128 __deltaTime = _mm_load_ps1( &deltaTime );
		const __m128 __particleMass = _mm_load1_ps( &particleMass );

		//float_3 pos(particleOut.pos);
		//float_3 vel(particleOut.vel);
		//float_3 acc(0.0f);
		__m128 pos = _mm_loadu_ps( (float*)&particleOut.pos );
		__m128 vel = _mm_loadu_ps( (float*)&particleOut.vel );
		__m128 acc = _mm_setzero_ps();

		// Cannot use lambdas here because __m128 is aligned.
		for( int j = 0; j < numParticles; ++j )
		{
			//float_3 r = p.pos - pos;
			__m128 pos1 = _mm_loadu_ps( (float*)&pParticlesIn[ j ].pos );
			__m128 r = _mm_sub_ps( pos1, pos );

			//float distSqr = float_3::SqrLength(r) + m_softeningSquared;
			__m128 distSqr = _mm_mul_ps( r, r );    //x    y    z    ?
			__m128 rshuf = _mm_shuffle_ps( distSqr, distSqr, _MM_SHUFFLE( 0, 3, 2, 1 ) );
			distSqr = _mm_add_ps( distSqr, rshuf );  //x+y, y+z, z+?, ?+x
			rshuf = _mm_shuffle_ps( distSqr, distSqr, _MM_SHUFFLE( 1, 0, 3, 2 ) );
			distSqr = _mm_add_ps( rshuf, distSqr );  //x+y+z+0, y+z+0+X, z+0+x+y, 0+x+y+z
			distSqr = _mm_add_ps( distSqr, __softeningSquared );

			//float invDist = 1.0f / sqrt(distSqr);
			//float invDistCube =  invDist * invDist * invDist;
			//float s = m_particleMass * invDistCube;
			__m128 invDistSqr = _mm_rsqrt_ps( distSqr );
			__m128 invDistCube = _mm_mul_ps( _mm_mul_ps( invDistSqr, invDistSqr ), invDistSqr );
			__m128 s = _mm_mul_ps( __particleMass, invDistCube );

			//acc += r * s;
			acc = _mm_add_ps( _mm_mul_ps( r, s ), acc );
		}

		//vel += acc * m_deltaTime;
		vel = _mm_add_ps( _mm_mul_ps( acc, __deltaTime ), vel );

		//vel *= m_dampingFactor;   
		vel = _mm_mul_ps( vel, __dampingFactor );

		//pos += vel * m_deltaTime;
		pos = _mm_add_ps( _mm_mul_ps( vel, __deltaTime ), pos );

		// The r3 word in each register has an undefined value at this point but
		// this isn't used elsewhere so there is no need to clear it.
		//particleOut.pos = pos;
		//particleOut.vel = vel;
		_mm_storeu_ps( (float*)&particleOut.pos, pos );
		_mm_storeu_ps( (float*)&particleOut.vel, vel );
	}

	void bodyBodyInteractionSSE4( const ParticleCpu* const pParticlesIn, ParticleCpu& particleOut, int numParticles, float deltaTime )
	{
		const __m128 __softeningSquared = _mm_load1_ps( &softeningSquared );
		const __m128 __dampingFactor = _mm_load_ps1( &dampingFactor );
		const __m128 __deltaTime = _mm_load_ps1( &deltaTime );
		const __m128 __particleMass = _mm_load1_ps( &particleMass );

		//float_3 pos(particleOut.pos);
		//float_3 vel(particleOut.vel);
		//float_3 acc(0.0f);
		__m128 pos = _mm_loadu_ps( (float*)&particleOut.pos );
		__m128 vel = _mm_loadu_ps( (float*)&particleOut.vel );
		__m128 acc = _mm_setzero_ps();

		// Cannot use lambdas here because __m128 is aligned.
		for( int j = 0; j < numParticles; ++j )
		{
			//float_3 r = p.pos - pos;
			__m128 pos1 = _mm_loadu_ps( (float*)&pParticlesIn[ j ].pos );
			__m128 r = _mm_sub_ps( pos1, pos );

			//float distSqr = float_3::SqrLength(r) + m_softeningSquared;
			//This uses the additional SSE4 _mm_dp_ps intrinsic.
			__m128 distSqr = _mm_dp_ps( r, r, 0x7F );
			distSqr = _mm_add_ps( distSqr, __softeningSquared );

			//float invDist = 1.0f / sqrt(distSqr);
			//float invDistCube =  invDist * invDist * invDist;
			//float s = m_particleMass * invDistCube;
			__m128 invDistSqr = _mm_rsqrt_ps( distSqr );
			__m128 invDistCube = _mm_mul_ps( _mm_mul_ps( invDistSqr, invDistSqr ), invDistSqr );
			__m128 s = _mm_mul_ps( __particleMass, invDistCube );

			//acc += r * s;
			acc = _mm_add_ps( _mm_mul_ps( r, s ), acc );
		}

		//vel += acc * m_deltaTime;
		vel = _mm_add_ps( _mm_mul_ps( acc, __deltaTime ), vel );

		//vel *= m_dampingFactor;
		vel = _mm_mul_ps( vel, __dampingFactor );

		//pos += vel * m_deltaTime;
		pos = _mm_add_ps( _mm_mul_ps( vel, __deltaTime ), pos );

		// The r3 word in each register has an undefined value at this point but
		// this isn't used elsewhere so there is no need to clear it.
		//particleOut.pos = pos;
		//particleOut.vel = vel;
		_mm_storeu_ps( (float*)&particleOut.pos, pos );
		_mm_storeu_ps( (float*)&particleOut.vel, vel );
	}

	//  Get the level of SSE support available on the current hardware. 
	inline CpuSSE getSSEType()
	{
		int CpuInfo[ 4 ] = { -1 };
		__cpuid( CpuInfo, 1 );

		if( CpuInfo[ 2 ] >> 19 & 0x1 ) return CpuSSE::CpuSSE4;
		if( CpuInfo[ 3 ] >> 24 & 0x1 ) return CpuSSE::CpuSSE;
		return CpuSSE::CpuNone;
	}
}

NBodySimpleInteractionEngine::NBodySimpleInteractionEngine()
{
	switch( getSSEType() )
	{
	case CpuSSE::CpuSSE4:
		_function = bodyBodyInteractionSSE4;
		break;
	case CpuSSE::CpuSSE:
		_function = bodyBodyInteractionSSE;
		break;
	default:
		_function = bodyBodyInteraction;
	}
}

void NBodySimpleInteractionEngine::InvokeBodyBodyInteraction( const ParticleCpu* const particlesIn, ParticleCpu& particleOut, int numParticles, float deltaTime ) const
{
	_function( particlesIn, particleOut, numParticles, deltaTime );
}

NBodySimpleSingleCore::NBodySimpleSingleCore()
	: _engine( std::make_unique< NBodySimpleInteractionEngine >() )
{}

void NBodySimpleSingleCore::integrate( ParticleCpu* const particlesIn, ParticleCpu*const particleOut, int numParticles, float deltaTime ) const
{
	for( int i = 0; i < numParticles; ++i )
	{
		particleOut[ i ] = particlesIn[ i ];
		_engine->InvokeBodyBodyInteraction( particlesIn, particleOut[ i ], numParticles, deltaTime );
	}
}

NBodySimpleMultiCore::NBodySimpleMultiCore()
	: _engine( std::make_unique< NBodySimpleInteractionEngine >() )
{}

void NBodySimpleMultiCore::integrate( ParticleCpu* const particlesIn, ParticleCpu*const particleOut, int numParticles, float deltaTime ) const
{
	parallel_for( 0, numParticles, [ =, &particleOut ]( int i )
	{
		particleOut[ i ] = particlesIn[ i ];
		_engine->InvokeBodyBodyInteraction( particlesIn, particleOut[ i ], numParticles, deltaTime );
	} );
}