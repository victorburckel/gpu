#include "../utils.h"
#include <tuple>
#include <vector>

__global__ void minmax_reduce_kernel( const float* const d_logLuminanceMin, const float* const d_logLuminanceMax, float* const d_min, float* const d_max )
{
	extern __shared__ float s_logLuminance[];

	const int index = threadIdx.x + blockDim.x * blockIdx.x;
	const int i = threadIdx.x;

	float* const s_logLuminanceMin = s_logLuminance;
	float* const s_logLuminanceMax = s_logLuminance + blockDim.x;

	s_logLuminanceMin[ i ] = d_logLuminanceMin[ index ];
	s_logLuminanceMax[ i ] = d_logLuminanceMax[ index ];

	__syncthreads();

	for( unsigned s = blockDim.x / 2; s > 0; s >>= 1 )
	{
		if( i < s )
		{
			s_logLuminanceMin[ i ] = fminf( s_logLuminanceMin[ i ], s_logLuminanceMin[ i + s ] );
			s_logLuminanceMax[ i ] = fmaxf( s_logLuminanceMax[ i ], s_logLuminanceMax[ i + s ] );
			__syncthreads();
		}
	}

	if( i == 0 )
	{
		d_min[ blockIdx.x ] = s_logLuminanceMin[ 0 ];
		d_max[ blockIdx.x ] = s_logLuminanceMax[ 0 ];
	}
}

std::pair< float, float > minmax_reduce( const float* const d_logLuminance, const size_t size )
{
	const int maxThreadsPerBlock = 1024;

	int block_dim = maxThreadsPerBlock;
	int grid_dim = size / block_dim;

	float* d_minmax;
	checkCudaErrors( cudaMalloc( &d_minmax, 2 * sizeof( float ) * grid_dim ) );

	minmax_reduce_kernel<<< grid_dim, block_dim, 2 * sizeof( float ) * block_dim >>>( d_logLuminance, d_logLuminance, d_minmax, d_minmax + grid_dim );

	cudaDeviceSynchronize();
	checkCudaErrors( cudaGetLastError() );

	block_dim = grid_dim;
	grid_dim = 1;

	minmax_reduce_kernel<<< grid_dim, block_dim, 2 * sizeof( float ) * block_dim >>>( d_minmax, d_minmax + block_dim, d_minmax, d_minmax + block_dim );

	cudaDeviceSynchronize();
	checkCudaErrors( cudaGetLastError() );

	std::pair< float, float > result;

	checkCudaErrors( cudaMemcpy( &result.first, d_minmax, sizeof( float ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy( &result.second, d_minmax + block_dim, sizeof( float ), cudaMemcpyDeviceToHost ) );

	checkCudaErrors( cudaFree( d_minmax ) );

	return result;
}

__global__ void computeHisto_kernel( const float* const d_logLuminance, unsigned int* const d_cdf, float min_logLum, float lumRange, size_t numBins )
{
	const int index = threadIdx.x + blockDim.x * blockIdx.x;
	
	float logLuminance = d_logLuminance[ index ];

	int bin = ( logLuminance - min_logLum ) / lumRange * numBins;

	atomicAdd( &( d_cdf[ bin ] ), 1 );
}

__global__ void hillisSteele_scan_kernel( unsigned int* const d_cdf, const unsigned int numBins )
{
	extern __shared__ int s_cdf[];

	int i = threadIdx.x;
	s_cdf[ i ] = d_cdf[ i ];
	
	__syncthreads();

	for( unsigned offset = 1; offset < numBins; offset <<= 1 )
	{
		if( i >= offset )
		{
			int temp = s_cdf[ i ] + s_cdf[ i - offset ];
			__syncthreads();

			s_cdf[ i ] = temp;
			__syncthreads();
		}
	}

	if( i != numBins - 1 )
		d_cdf[ i + 1 ] = s_cdf[ i ];
	else
		d_cdf[ 0 ] = 0;
}

int main()
{
	const size_t numCols = 1024;
	const size_t numRows = 768;
	const size_t numBins = 1024;

	std::vector< float > h_logLuminance( numCols * numRows );

	for( unsigned int i = 0; i < numCols * numRows; ++i )
		h_logLuminance[ i ] = i % 100 + 10;

	float min_logLum;
	float max_logLum;

	float* d_logLuminance;
	unsigned int* d_cdf;
	
	checkCudaErrors( cudaMalloc( &d_logLuminance, sizeof( float ) * numRows * numCols ) );
	checkCudaErrors( cudaMalloc( &d_cdf, sizeof( int ) * numBins ) );
	checkCudaErrors( cudaMemcpy( d_logLuminance, &h_logLuminance[ 0 ], sizeof( float ) * numRows * numCols, cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemset( d_cdf, 0, sizeof( int ) * numBins ) );

	// 1) find the minimum and maximum value in the input logLuminance channel store in min_logLum and max_logLum
	std::tie( min_logLum, max_logLum ) = minmax_reduce( d_logLuminance, numCols * numRows );

	// 2) subtract them to find the range
	float lumRange = max_logLum - min_logLum;

	// 3) generate a histogram of all the values in the logLuminance channel using	the formula : bin = ( lum[ i ] - lumMin ) / lumRange * numBins
	computeHisto_kernel<<< numCols * numRows / 1024, 1024 >>>( d_logLuminance, d_cdf, min_logLum, lumRange, numBins );

	cudaDeviceSynchronize();
	checkCudaErrors( cudaGetLastError() );

	// 4) Perform an exclusive scan (prefix sum) on the histogram to get the cumulative distribution of luminance values( this should go in the incoming d_cdf pointer which already has been allocated for you )

	/*std::vector< unsigned int > h_cdf( numBins );
	for( int i = 0; i < numBins; ++i )
		h_cdf[ i ] = i % 5 + 1*/;

	/*checkCudaErrors( cudaMemcpy( d_cdf, &h_cdf[ 0 ], sizeof( unsigned int ) * numBins, cudaMemcpyHostToDevice ) );*/

	hillisSteele_scan_kernel<<< 1, numBins, sizeof( int ) * numBins >>>( d_cdf, numBins );
	
	cudaDeviceSynchronize();
	checkCudaErrors( cudaGetLastError() );

	/*checkCudaErrors( cudaMemcpy( &h_cdf[ 0 ], d_cdf, sizeof( unsigned int ) * numBins, cudaMemcpyDeviceToHost ) );*/

	checkCudaErrors( cudaFree( d_cdf ) );
	checkCudaErrors( cudaFree( d_logLuminance ) );
	

	return 0;
}