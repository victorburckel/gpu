#pragma once

#include <amp.h>
#include <amp_short_vectors.h>
#include <vector>

struct ParticlesCpu
{
	std::vector< concurrency::graphics::float_3 > pos;
	std::vector< concurrency::graphics::float_3 > vel;

	ParticlesCpu( int size ) : pos( size ), vel( size )
	{}

	inline int size() const
	{
		assert( pos.size() == vel.size() );
		return static_cast<int>( pos.size() );
	}
};

struct ParticlesAmp
{
	concurrency::array< concurrency::graphics::float_3, 1 >& pos;
	concurrency::array< concurrency::graphics::float_3, 1 >& vel;

public:
	ParticlesAmp( concurrency::array< concurrency::graphics::float_3, 1 >& pos, concurrency::array< concurrency::graphics::float_3, 1 >& vel ) : pos( pos ), vel( vel )
	{}

	inline int size() const
	{
		return pos.extent.size();
	}
};

struct TaskData
{
public:
	concurrency::accelerator accelerator;
	std::unique_ptr< ParticlesAmp > dataOld;
	std::unique_ptr< ParticlesAmp > dataNew;

private:
	concurrency::array< concurrency::graphics::float_3, 1 > _posOld;
	concurrency::array< concurrency::graphics::float_3, 1 > _posNew;
	concurrency::array< concurrency::graphics::float_3, 1 > _velOld;
	concurrency::array< concurrency::graphics::float_3, 1 > _velNew;

public:
	TaskData( int size, concurrency::accelerator_view view, concurrency::accelerator acc ) :
		accelerator( acc ),
		_posOld( size, view ),
		_velOld( size, view ),
		_posNew( size, view ),
		_velNew( size, view ),
		dataOld( std::make_unique< ParticlesAmp >( _posOld, _velOld ) ),
		dataNew( std::make_unique< ParticlesAmp >( _posNew, _velNew ) )
	{}
};
