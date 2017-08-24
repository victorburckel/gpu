#pragma once

#include <cuda_runtime.h>

class GpuTimer
{
public:
	GpuTimer()
	{
		cudaEventCreate( &_start );
		cudaEventCreate( &_stop );
	}
	~GpuTimer()
	{
		cudaEventDestroy( _start );
		cudaEventDestroy( _stop );
	}
	void start()
	{
		cudaEventRecord( _start, 0 );
	}
	void stop()
	{
		cudaEventRecord( _stop, 0 );
	}

	float elapsed()
	{
		float elapsed;
		cudaEventSynchronize( _stop );
		cudaEventElapsedTime( &elapsed, _start, _stop );
		return elapsed;
	}

private:
	cudaEvent_t _start;
	cudaEvent_t _stop;
};