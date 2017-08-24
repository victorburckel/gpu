//Udacity HW 4
//Radix Sorting

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */


#include "../utils.h"
#include "../timer.h"
#include "../scan.h"
#include "../reduce.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <thrust/functional.h>

const size_t numBits = 4;
const size_t numBins = 1 << numBits;

__global__ void temp_kernel( const unsigned int* const d_inputVals, unsigned int* const d_temp, const unsigned shift, const unsigned int size )
{
	const int index = threadIdx.x + blockIdx.x * blockDim.x;

	if( index >= size )
		return;

	const unsigned int value = d_inputVals[ index ];
	const unsigned int mask = ( numBins - 1 );

	const unsigned int bin = ( value >> shift ) & mask;

	d_temp[ index + size * bin ] = 1;
}

__global__ void scatter_kernel( const unsigned int* const d_inputVals, const unsigned int* const d_intputPos, const unsigned int* const d_temp, unsigned int* d_outputVals, unsigned int* d_outputPos, const unsigned shift, const unsigned int size )
{
	const int index = threadIdx.x + blockIdx.x * blockDim.x;

	if( index >= size )
		return;

	const unsigned int value = d_inputVals[ index ];
	const unsigned int mask = ( numBins - 1 );

	const unsigned int bin = ( value >> shift ) & mask;

	const unsigned int pos = d_temp[ index + size * bin ];

	d_outputVals[ pos ] = value;
	d_outputPos[ pos ] = d_intputPos[ index ];
}

void your_sort( unsigned int* const d_inputVals,
				unsigned int* const d_inputPos,
				unsigned int* const d_outputVals,
				unsigned int* const d_outputPos,
				const size_t numElems )
{
	unsigned int* d_vals_src = d_inputVals;
	unsigned int* d_pos_src = d_inputPos;
	unsigned int* d_vals_dst = d_outputVals;
	unsigned int* d_pos_dst = d_outputPos;

	unsigned int* d_temp;

	checkCudaErrors( cudaMalloc( &d_temp, sizeof( unsigned int ) * numElems * numBins ) );

	const unsigned int bloc_dim = std::min( 1024u, numElems );
	const unsigned int grid_dim = numElems / bloc_dim + ( numElems % bloc_dim ? 1u : 0u );;

	for( unsigned int shift = 0; shift < 8 * sizeof( unsigned int ); shift += numBits )
	{
		checkCudaErrors( cudaMemset( d_temp, 0, sizeof( unsigned int ) * numElems * numBins ) );

		temp_kernel << < grid_dim, bloc_dim >> >( d_vals_src, d_temp, shift, numElems );

		cudaDeviceSynchronize();
		checkCudaErrors( cudaGetLastError() );

		blelloch_scan( d_temp, d_temp, numElems * numBins, thrust::plus< unsigned int >(), 0u );

		scatter_kernel << < grid_dim, bloc_dim >> >( d_vals_src, d_pos_src, d_temp, d_vals_dst, d_pos_dst, shift, numElems );

		cudaDeviceSynchronize();
		checkCudaErrors( cudaGetLastError() );

		std::swap( d_vals_src, d_vals_dst );
		std::swap( d_pos_src, d_pos_dst );
	}

	if( d_vals_src != d_outputVals )
	{
		checkCudaErrors( cudaMemcpy( d_outputVals, d_vals_src, sizeof( unsigned int ) * numElems, cudaMemcpyDeviceToDevice ) );
		checkCudaErrors( cudaMemcpy( d_outputPos, d_pos_src, sizeof( unsigned int ) * numElems, cudaMemcpyDeviceToDevice ) );
	}

	checkCudaErrors( cudaFree( d_temp ) );
}
