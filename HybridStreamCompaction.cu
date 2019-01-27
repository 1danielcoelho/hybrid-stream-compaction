#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <string.h>
#include <iostream>
#include <fstream>
#include <iostream>
#include <algorithm>

#define FULL_MASK 0xFFFFFFFF
#define WARP_SIZE 32
#define SUBGROUP_SIZE 32
#define GROUP_SIZE 1024

using namespace std;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

#define eee(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void hybridKernel(uchar* d_inData, uchar* d_outData, uchar M, uint* d_globalOffset)
{
	uint groupID = (blockIdx.x * blockDim.x + threadIdx.x) / GROUP_SIZE;  // which 1024 element group we're sweeping
	uint laneID = threadIdx.x % WARP_SIZE;  // [0,32] index of this thread within its warp						
	uint votes;  // Which threads in the warp passed the predicate for subgroupID == laneID	
	uint cnt;  // How many threads in the warp passed the predicate for subgroupID == laneID
	
	// Stage 1
	#pragma unroll
	for (uint subgroupID = 0; subgroupID < SUBGROUP_SIZE; subgroupID++)
	{
		uint index = groupID * GROUP_SIZE + subgroupID * SUBGROUP_SIZE + laneID;		

		uint votesLocal = __ballot_sync(FULL_MASK, d_inData[index] > M);
		uint cntLocal = __popc(votesLocal);

		if (subgroupID == laneID)
		{	
			votes = votesLocal;
			cnt = cntLocal;			
		}
	}

	// Stage 2
	// Perform a parallel scan of 'cnt' across subgroups, i.e. after this
	// cnt describes a sum of how many threads in the warp passed the predicate for subgroupID <= laneID
	// laneID 31's cnt will describe how many threads in the the group passed the predicate
	#pragma unroll
	for (uint delta = 1; delta < SUBGROUP_SIZE; delta *= 2)
	{		
		uint cntFromLowerLane = __shfl_up_sync(FULL_MASK, cnt, delta);
		if (laneID >= delta)
		{
			cnt += cntFromLowerLane;
		}
	}

	// Stage 3
	// Gets global offset of the entire group and send it to every thread in the warp
	uint groupOffset = 0;
	if (laneID == WARP_SIZE-1)
	{
		groupOffset = atomicAdd(d_globalOffset, cnt);			
	}
	groupOffset = __shfl_sync(FULL_MASK, groupOffset, WARP_SIZE-1);

	// Stage 4
	#pragma unroll
	for (uint subgroupID = 0; subgroupID < SUBGROUP_SIZE; subgroupID++)
	{
		uint subgroupOffset = (subgroupID == 0)? 0 : __shfl_sync(FULL_MASK, cnt, subgroupID-1);

		// our 'cnt' is now a scan, so we need to do __popc again
		// The bit mask selects only bits that represent threads with a lower lane ID than this one
		uint votesLocal = __shfl_sync(FULL_MASK, votes, subgroupID);		
		uint intraSubgroupOffset = __popc(votesLocal & ((1 << laneID)-1)); 		

		uint selectedIndex = groupOffset + subgroupOffset + intraSubgroupOffset;
		
		if (votesLocal & (1 << laneID))
		{
			d_outData[selectedIndex] = d_inData[groupID * GROUP_SIZE + subgroupID * SUBGROUP_SIZE + laneID];
		}
	}
}

uint runCUDA(uchar* h_inData, uchar* h_outData, uint numElements, uchar M)
{
	size_t numBytes = numElements * sizeof(uchar);

	uchar* d_inData;
	eee(cudaMalloc((void **) &d_inData, numBytes));
	eee(cudaMemcpy(d_inData, h_inData, numBytes, cudaMemcpyHostToDevice));	

	uchar* d_outData;
	eee(cudaMalloc((void **) &d_outData, numBytes));
	
	uint* d_globalOffset;
	eee(cudaMalloc((void **) &d_globalOffset, 1 * sizeof(uint)));
	eee(cudaMemset((void*) d_globalOffset, 0, 1 * sizeof(uint)));
	
	// Find largest threadsPerBlock number that evenly splits numElements
	uint threadsPerBlock = 1024;
	uint elementsPerBlock = threadsPerBlock * SUBGROUP_SIZE;
	while (numElements % elementsPerBlock != 0)
	{
		threadsPerBlock /= 2;
		elementsPerBlock = threadsPerBlock * SUBGROUP_SIZE;
		
		assert(threadsPerBlock >= 32);
	}	
	
	dim3 grid(numElements / elementsPerBlock, 1, 1);
	dim3 threads(threadsPerBlock, 1, 1);

	printf("numElements = %u: using %u blocks, with %u threadsPerBlock\n", numElements, (numElements/elementsPerBlock), threadsPerBlock); 

	auto start = std::chrono::high_resolution_clock::now();
	{
		hybridKernel<<<grid, threads>>>(d_inData, d_outData, M, d_globalOffset);    
		eee(cudaDeviceSynchronize()); // Make sure we complete the kernel before getting the timer result
	}
	auto duration = std::chrono::high_resolution_clock::now() - start;
	long long ms = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
	printf("runCUDA executed in %lld microseconds\n", ms);    
    
	eee(cudaGetLastError());
	eee(cudaMemcpy(h_outData, d_outData, numBytes, cudaMemcpyDeviceToHost)); 
	
	uint h_globalOffset = 0;
	eee(cudaMemcpy(&h_globalOffset, d_globalOffset, 1*sizeof(uint), cudaMemcpyDeviceToHost));    
    
	eee(cudaFree(d_inData));
	eee(cudaFree(d_outData));	
	eee(cudaFree(d_globalOffset));
	return h_globalOffset;
}

int main(int argc, char **argv)
{
	cout << "Starting" << endl;
	
	uint numElements = 1024 * 125000;
	vector<uchar> inputData(numElements);
	for (uint i = 0; i < numElements; i++)
	{
		inputData[i] = (uchar)((rand() * 255.0f)/RAND_MAX + 0.5f);
	}

	// Our 'predicate' will be that the element be larger than M
	uchar M = 126;
	vector<uchar> largerThanM(numElements);

	uint numLargerThanM = runCUDA(inputData.data(), largerThanM.data(), numElements, M);
	
	uchar minElement = *std::min_element(largerThanM.begin(), largerThanM.begin() + numLargerThanM);

	printf("numLargerThanM: %u, minElement: %u\n", numLargerThanM, minElement);	
}