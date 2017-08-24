#pragma once

struct ParticleCpu;

struct INBodyCpu
{
	virtual void integrate( ParticleCpu* const pParticlesIn, ParticleCpu* const pParticlesOut, int numParticles, float deltaTime ) const = 0;
};