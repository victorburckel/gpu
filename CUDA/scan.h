#pragma once

#include "utils.h"
#include <algorithm>

template< typename T, typename Op >
__global__ void hillis_steele_scan_kernel( const T* const d_data, T* const d_scan, const Op op )
{
	extern __shared__ T s_data[];

	const int index = threadIdx.x + blockIdx.x * blockDim.x;
	const int i = threadIdx.x;

	s_data[ i ] = d_data[ index ];

	__syncthreads();

	for( unsigned s = 1; s < blockDim.x; s <<= 1 )
	{
		int temp;
		if( i >= s )
			temp = op( s_data[ i ], s_data[ i - s ] );

		__syncthreads();

		if( i >= s )
			s_data[ i ] = temp;
		
		__syncthreads();
	}

	d_scan[ index ] = s_data[ i ];
}

template< typename T >
__global__ void hillis_steele_scan_extract_kernel( const T* const d_scan, T* const d_temp, const unsigned int block_dim )
{
	const int i = threadIdx.x;

	d_temp[ i ] = d_scan[ ( i + 1 ) * block_dim - 1 ];
}

template< typename T >
__global__ void hillis_steele_scan_add_kernel( T* const d_scan, const T* const d_temp )
{
	const int index = threadIdx.x + ( 1 + blockIdx.x ) * blockDim.x;

	d_scan[ index ] += d_temp[ blockIdx.x ];
}


// Inclusive Hillis Steele scan
template< typename T, typename Op >
void hillis_steele_scan( const T* const d_data, T* const d_scan, const size_t size, const Op op )
{
	const size_t maxThreadsPerBlock = 512u;

	const size_t block_dim = std::min( size, maxThreadsPerBlock );
	const size_t grid_dim = size / block_dim;

	hillis_steele_scan_kernel << < grid_dim, block_dim, sizeof( T ) * block_dim >> >( d_data, d_scan, op );

	cudaDeviceSynchronize();
	checkCudaErrors( cudaGetLastError() );

	if( grid_dim > 1 )
	{
		T* d_temp;
		checkCudaErrors( cudaMalloc( &d_temp, sizeof( T ) * ( grid_dim - 1 ) ) );

		hillis_steele_scan_extract_kernel << < 1, grid_dim - 1 >> >( d_scan, d_temp, block_dim );

		cudaDeviceSynchronize();
		checkCudaErrors( cudaGetLastError() );

		hillis_steele_scan_kernel << < 1, grid_dim - 1, sizeof( T ) * ( grid_dim - 1 ) >> >( d_temp, d_temp, op );

		cudaDeviceSynchronize();
		checkCudaErrors( cudaGetLastError() );

		hillis_steele_scan_add_kernel << < grid_dim - 1, block_dim >> >( d_scan, d_temp );

		cudaDeviceSynchronize();
		checkCudaErrors( cudaGetLastError() );

		checkCudaErrors( cudaFree( d_temp ) );
	}
}

template< typename T, typename Op >
__global__ void blelloch_scan_kernel( const T* const d_data, T* const d_scan, const Op op, const T seed )
{
	extern __shared__ T s_data[];

	const int index = threadIdx.x + blockIdx.x * blockDim.x;
	const int i = threadIdx.x;

	s_data[ 2 * i ] = d_data[ 2 * index ];
	s_data[ 2 * i + 1 ] = d_data[ 2 * index + 1 ];

	// Reduce
	unsigned int size = 0u;
	for( unsigned s = blockDim.x; s > 1; size += 2 * s, s >>= 1 )
	{
		if( i < s )
			s_data[ size + 2 * s + i ] = s_data[ size + 2 * i ] + s_data[ size + 2 * i + 1 ];

		__syncthreads();
	}

	if( i == 0 )
		s_data[ size + 2 ] = seed;

	// DownSweep
	for( unsigned s = 2; s <= 2 * blockDim.x; size -= 2 * s, s <<= 1 )
	{
		if( i < s / 2 )
		{
			const unsigned sweep = s_data[ size + s + i ];

			s_data[ size + 2 * i + 1 ] = s_data[ size + 2 * i ] + sweep;
			s_data[ size + 2 * i ] = sweep;
		}

		__syncthreads();
	}

	d_scan[ 2 * index ] = s_data[ 2 * i ];
	d_scan[ 2 * index + 1 ] = s_data[ 2 * i + 1 ];
}

template< typename T >
__global__ void blelloch_scan_pre_extract_kernel( const T* const d_data, T* const d_temp, const unsigned int block_dim, const unsigned int size )
{
	for( unsigned int index = threadIdx.x + blockIdx.x * blockDim.x; index < size; index+= blockIdx.x * blockDim.x )
		d_temp[ index ] = d_data[ ( index + 1 ) * block_dim - 1 ];
}

template< typename T >
__global__ void blelloch_scan_extract_kernel( const T* const d_scan, T* const d_temp, const unsigned int block_dim )
{
	const int index = threadIdx.x + blockIdx.x * blockDim.x;

	d_temp[ index ] += d_scan[ ( index + 1 ) * block_dim - 1 ];
}

size_t getGridSize( const size_t size, const size_t blockSize )
{
	return ( size / blockSize ) + ( size % blockSize ? 1u : 0u );
}

// Exclusive Blelloch scan
template< typename T, typename Op >
void blelloch_scan( const T* const d_data, T* const d_scan, const size_t size, const Op op, const T seed )
{
	const size_t maxThreadsPerBlock = 512u;

	const size_t block_dim = std::min( size / 2, maxThreadsPerBlock );
	const size_t grid_dim = getGridSize( size, 2 * block_dim );

	T* d_temp;
	if( grid_dim > 1 )
	{
		checkCudaErrors( cudaMalloc( &d_temp, sizeof( T ) * ( grid_dim - 1 ) ) );

		blelloch_scan_pre_extract_kernel << < 1, grid_dim - 1 >> >( d_data, d_temp, block_dim * 2 );

		cudaDeviceSynchronize();
		checkCudaErrors( cudaGetLastError() );
	}

	blelloch_scan_kernel << < grid_dim, block_dim, sizeof( T ) * block_dim * 4 >> >( d_data, d_scan, op, seed );

	cudaDeviceSynchronize();
	checkCudaErrors( cudaGetLastError() );

	if( grid_dim > 1 )
	{
		blelloch_scan_extract_kernel << < 1, grid_dim - 1 >> >( d_scan, d_temp, block_dim * 2 );

		cudaDeviceSynchronize();
		checkCudaErrors( cudaGetLastError() );

		hillis_steele_scan_kernel << < 1, grid_dim - 1, sizeof( T ) * ( grid_dim - 1 ) >> >( d_temp, d_temp, op );

		cudaDeviceSynchronize();
		checkCudaErrors( cudaGetLastError() );

		hillis_steele_scan_add_kernel << < grid_dim - 1, block_dim * 2 >> >( d_scan, d_temp );

		cudaDeviceSynchronize();
		checkCudaErrors( cudaGetLastError() );

		checkCudaErrors( cudaFree( d_temp ) );
	}
}