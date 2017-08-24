#pragma once

#include "utils.h"
#include <cassert>
#include <algorithm>

template< typename T, typename Op >
__global__ void reduce_kernel( const T* const d_data, T* const d_reduce, const Op op )
{
	extern __shared__ T s_data[];

	const int index = threadIdx.x + blockDim.x * blockIdx.x;
	const int i = threadIdx.x;

	s_data[ i ] = d_data[ index ];

	__syncthreads();

	for( unsigned s = blockDim.x / 2; s > 0; s >>= 1 )
	{
		if( i < s )
			s_data[ i ] = op( s_data[ i ], s_data[ i + s ] );
			
		__syncthreads();
	}

	if( i == 0 )
		d_reduce[ blockIdx.x ] = s_data[ 0 ];
}

bool is_power_2( size_t x )
{
	return ( x > 0 ) && ( ( x & ( x - 1 ) ) == 0 );
}

template< typename T, typename Op >
T reduce( const T* const d_data, const size_t size, const Op op )
{
	const size_t maxThreadsPerBlock = 1024u;

	const size_t block_dim = std::min( size, maxThreadsPerBlock );
	const size_t grid_dim = size / block_dim;

	// block_dim needs to be a power of two
	assert( is_power_2( block_dim ) );

	T* d_reduce;
	checkCudaErrors( cudaMalloc( &d_reduce, sizeof( T ) * grid_dim ) );

	reduce_kernel << < grid_dim, block_dim, sizeof( T ) * block_dim >> >( d_data, d_reduce, op );

	cudaDeviceSynchronize();
	checkCudaErrors( cudaGetLastError() );

	if( grid_dim > 1 )
	{
		assert( is_power_2( grid_dim ) );

		reduce_kernel << < 1, grid_dim, sizeof( T ) * grid_dim >> >( d_reduce, d_reduce, op );

		cudaDeviceSynchronize();
		checkCudaErrors( cudaGetLastError() );
	}

	T result;
	checkCudaErrors( cudaMemcpy( &result, d_reduce, sizeof( T ), cudaMemcpyDeviceToHost ) );

	checkCudaErrors( cudaFree( d_reduce ) );

	return result;
}