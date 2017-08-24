#pragma once

#include <amp_short_vectors.h>

#define SSE_ALIGNMENTBOUNDARY 16

__declspec( align( SSE_ALIGNMENTBOUNDARY ) )
struct ParticleCpu
{
	concurrency::graphics::float_3 pos;
	float ssePpadding1;
	concurrency::graphics::float_3 vel;
	float ssePpadding2;
	concurrency::graphics::float_3 acc;
	float ssePpadding3;
	concurrency::graphics::float_4 cacheLinePadding;
};

__declspec( align( SSE_ALIGNMENTBOUNDARY ) )
struct ParticleSSE
{
	__m128 pos;
	__m128 vel;
	__m128 acc;
	__m128 cacheLinePadding;
};
