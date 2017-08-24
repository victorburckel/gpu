#pragma once

#include "INBodyCpu.h"
#include <memory>
#include <functional>

//  User selected integration algorithm implementation.
enum struct ComputeType
{
	CpuSingle = 0,
	CpuMulti = 1,
	CpuAdvanced = 2
};

//  Level of SSE support available. Determined dynamically at runtime.
enum struct CpuSSE
{
	CpuNone = 0,
	CpuSSE,
	CpuSSE4
};

class NBodySimpleInteractionEngine
{
public:
	NBodySimpleInteractionEngine();
	void InvokeBodyBodyInteraction( const ParticleCpu* const particlesIn, ParticleCpu& particleOut, int numParticles, float deltaTime ) const;

private:
	std::function< void( const ParticleCpu* const pParticlesIn, ParticleCpu& particleOut, int numParticles, float deltaTime ) > _function;
};

class NBodySimpleSingleCore : public INBodyCpu
{
public:
	NBodySimpleSingleCore();
	virtual void integrate( ParticleCpu* const pParticlesIn, ParticleCpu*const pParticlesOut, int numParticles, float deltaTime ) const override;

private:
	std::unique_ptr<NBodySimpleInteractionEngine> _engine;
};

class NBodySimpleMultiCore : public INBodyCpu
{
public:
	NBodySimpleMultiCore();
	virtual void integrate( ParticleCpu* const pParticlesIn, ParticleCpu*const pParticlesOut, int numParticles, float deltaTime ) const override;

private:
	std::unique_ptr<NBodySimpleInteractionEngine> _engine;
};