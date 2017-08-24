#include "NBodyAMP.h"
#include "ParticlesAMP.h"
#include <amp_math.h>

using namespace concurrency;
using namespace concurrency::graphics;

namespace {
	const float softeningSquared = 0.0000015625f;
	const float dampingFactor = 0.9995f;
	const float particleMass = ( ( 6.67300e-11f * 10000.0f ) * 10000.0f * 10000.0f );

	inline const float sqrLength( const float_3& r ) restrict( amp )
	{
		return r.x * r.x + r.y * r.y + r.z * r.z;
	}

	void bodyBodyInteraction( float_3& acc, const float_3 particlePosition, const float_3 otherParticlePosition, float softeningSquared, float particleMass ) restrict( amp )
	{
		float_3 r = otherParticlePosition - particlePosition;

		float distSqr = sqrLength( r ) + softeningSquared;
		float invDist = fast_math::rsqrt( distSqr );
		float invDistCube = invDist * invDist * invDist;

		float s = particleMass * invDistCube;

		acc += r * s;
	}
}

int NBodyAMP::tileSize() const
{
	return 1;
}

void NBodyAMP::integrate( TaskData& particleData, int numParticles, float deltaTime ) const
{
	assert( numParticles > 0 );
	assert( ( numParticles % 4 ) == 0 );

	auto& particlesIn = *particleData.dataOld;
	auto& particlesOut = *particleData.dataNew;

	extent<1> computeDomain( numParticles );

	const auto __softeningSquared = softeningSquared;
	const auto __particleMass = particleMass;
	const auto __dampingFactor = dampingFactor;

	parallel_for_each( computeDomain, [ = ]( index<1> idx ) restrict( amp )
	{
		float_3 pos = particlesIn.pos[ idx ];
		float_3 vel = particlesIn.vel[ idx ];
		float_3 acc = 0.0f;

		// Update current Particle using all other particles
		for( int j = 0; j < numParticles; ++j )
			bodyBodyInteraction( acc, pos, particlesIn.pos[ j ], __softeningSquared, __particleMass );

		vel += acc * deltaTime;
		vel *= __dampingFactor;
		pos += vel * deltaTime;

		particlesOut.pos[ idx ] = pos;
		particlesOut.vel[ idx ] = vel;
	} );
}