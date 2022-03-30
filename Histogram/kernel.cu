//HW2 submission for James Fong.
//Corraborators with Kerim Karabacak and Matthew Lepis.
//All sources are either listed on this code, derived from the practice example in CUDA for Histograms and shared memory, or listed in the HW2 Lab Report.

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <helper_cuda.h>
#include <helper_functions.h> 

using namespace std;

// Histogram calculation pulled from CoffeeBeforeArch located here: https://www.youtube.com/watch?v=Bwv5J7dHYjU
// Git page located here: https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/04_histogram/shmem_atomic/histogram.cu

__global__ void histogram(int* d_input, int* d_bins, int N, int N_bins, int DIV) {
	// Allocate shared memory
	extern __shared__ int s_bins[];

	// Calculate a global thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Initialize our shared memory
	if (threadIdx.x < N_bins)
		s_bins[threadIdx.x] = 0; // every thread block will have its own version of s_bins

	// Synchronize; wait for threads to zero out shared memory
	__syncthreads(); //need to make sure no thread starts to accumulate in s_bins before it 0's out

	// Range check
	if (tid < N) {
		int bin = d_input[tid] / DIV;
		atomicAdd(&s_bins[bin], 1);
	}

	__syncthreads(); // make sure eveyrone has done their atomic adds, everyone in thread block has their shared memory, before finish binning their elements

	// Write back our partial results to main memory
	if (threadIdx.x < N_bins)
		atomicAdd(&d_bins[threadIdx.x], s_bins[threadIdx.x]);
}

//Initialize an array with random numbers, 1023 being the maximum
void init_array(int* input, int N, int MAX) {
	for (int i = 0; i < N; i++) {
		input[i] = rand() % MAX;
	}
}

int main(int argc, char* argv[]) {

	//Defining characters for argument
	StopWatchInterface* hTimer = NULL;
	int N;
	int N_bins;
	const int numRuns = 16;

	//Setting number of elements to bin (by default without input, will be 2^8)

	if (argc < 2) { // Not enough arguments for input

		cout << "No / not enough arguments given to command line, implementing default values elements = 2^12, bins = 10. \n\n" << endl;

		N = 1 << 10; // Set number of elements
		N_bins = 1 << 8; // set number of bins

	}

	else { // Enough arguments set for input

		int e = strtol(argv[1], nullptr, 0); // Exponent for number of elements
		int b = strtol(argv[2], nullptr, 0); // Number of bins

		cout << "Set number of bins to " << b << " elements. \n\n";
		cout << "Set vector dimension for " << e << " bin histogram. \n\n";

		N_bins = 1 << b; // Set number of bins by N_bins
		N = 1 << e; // Set number of elemnts to bins for 2^k

	}

	int Histogram_Bin_Count = N_bins; // For setting Histogram Bin Count

	//Set the number of elements to bin (2^12 default)
	size_t bytes = N * sizeof(int);

	// Set the number of bins, which should be written as 2^k where k can be any integer from 2 to 8
	size_t bytes_bins = N_bins * sizeof(int);

	sdkCreateTimer(&hTimer);

	//Setting input and bin values
	int* d_input, * d_bins,  // Device input and bins
		* h_HistogramGPU_input, * h_HistogramGPU_bins,  // Host input and bins
		* h_HistogramCPU_input, * h_HistogramCPU_bins; // CPU input and bins

	//Allocating memory for Host
	cudaMallocHost(&h_HistogramGPU_input, bytes);
	cudaMallocHost(&h_HistogramGPU_bins, bytes_bins);

	//Allocating memory for Device
	cudaMalloc(&d_input, bytes);
	cudaMalloc(&d_bins, bytes_bins);

	//Allocating memory for CPU
	cudaMallocHost(&h_HistogramCPU_input, bytes);
	cudaMallocHost(&h_HistogramCPU_bins, bytes_bins);

	// Set the max value for data as 1023 as instructed in HW
	int MAX = 1023;

	// Initialize our input data, pulled from CoffeeBeforeArch Example Histogram Shared Mem
	init_array(h_HistogramGPU_input, N, MAX); //init_array(input, N, MAX); 

	// Set the divisor, pulled from CoffeeBeforeArch Example Histogram Shared Mem
	int DIV = (MAX + N_bins - 1) / N_bins;

	// Set bins = 0 for first run instance
	for (int i = 0; i < N_bins; i++) {
		h_HistogramGPU_bins[i] = 0;
	}

	// Set the Cooperative Thread Array(thread blocks) and Grid Dimensions, pulled from CoffeeBeforeArch Example Histogram Shared Mem
	int THREADS = 1024; //for alternative testing, 512
	int BLOCKS = (N + THREADS - 1) / THREADS;

	// Setting size of dynamically allocated shared memory, pulled from CoffeeBeforeArch Example Histogram Shared Mem
	size_t SHMEM = N_bins * sizeof(int);

	// Beginning of kernel function call
	cout << "Starting up " << N_bins << "-bin histogram. \n\n";

	h_HistogramCPU_input = h_HistogramGPU_input;
	h_HistogramCPU_bins = h_HistogramGPU_bins;

	// CudaMemcpy from Host to Device
	cudaMemcpy(d_input, h_HistogramGPU_input, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bins, h_HistogramGPU_bins, bytes_bins, cudaMemcpyHostToDevice);

	// Kernel call, pulled from CoffeeBeforeArch
	histogram << <BLOCKS, THREADS, SHMEM >> > (d_input, d_bins, N, N_bins, DIV);

	// CudaMemcpy from Device to Host
	cudaMemcpy(h_HistogramGPU_input, d_input, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_HistogramGPU_bins, d_bins, bytes_bins, cudaMemcpyDeviceToHost);

	//prints out histogram data, pulled from histogram main.cpp
	int tmp = 0; //temporary value

	printf("Elements in bins: \n\n");

	for (int i = 0; i < N_bins; i++) {
		tmp += h_HistogramGPU_bins[i];
		cout << h_HistogramGPU_bins[i] << " ";
	}
	cout << "\n\n";
	cout << "Number of elements: \n\n" << tmp << "\n\n"; // Number of bins

	//Accessing kernel for histogram data, pulled from histogram main.cpp
	for (int iter = -1; iter < numRuns; iter++)
	{
		//iter == -1 -- warmup iteration
		if (iter == 0)
		{
			cudaDeviceSynchronize();
			sdkResetTimer(&hTimer);
			sdkStartTimer(&hTimer);
		}

		histogram << <BLOCKS, THREADS, SHMEM >> > (d_input, d_bins, N, N_bins, DIV);
	}

	// Compares Host data with CPU data if there is an error; pulled from example histogram main.cpp
	printf("Beginning Histogram data comparison with Host and Device: \n\n");

	for (unsigned int i = 0; i < Histogram_Bin_Count; i++)
		if (h_HistogramGPU_input[i] != h_HistogramCPU_input[i]) {
			printf("Failure; histogram comparison failed. Exiting program.");
			return 0;
		}

	printf("Histogram data comparison passed. \n\n");

	cudaDeviceSynchronize();
	sdkStopTimer(&hTimer); // Preparing timer

	int Histogram_ThreadBlock_Size = N_bins;

	//Histogram_ThreadBlock_Size is determined by Warp_Count * Warp_Size, dependent on histogram needed size and GPU compute capability.

	//Printing Timer run values...

	double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
	printf("histogram_gpu time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)bytes * 1.0e-6) / dAvgSecs);
	printf("histogram_gpu, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n\n",
		(1.0e-6 * (double)bytes / dAvgSecs), sdkGetTimerValue(&hTimer), bytes, 1, Histogram_ThreadBlock_Size);

	//Operations done to calculate total atomic operations. Operations were created with Kerim and Matt.
	int overflow = N % THREADS;
	int total_atomic_ops = N + ((((BLOCKS - 1) * THREADS) + overflow) * N_bins);

	cout << "Total atomic operations: " << total_atomic_ops << endl << endl;
	double MOAO = (double)total_atomic_ops / 1000000000;
	double MAOPS = MOAO / dAvgSecs;


	// Closing CUDA and freeing memory
	printf("Shutting down...\n");
	sdkDeleteTimer(&hTimer);

	cudaFree(d_input);
	cudaFree(d_bins);
	cudaFreeHost(h_HistogramGPU_input);
	cudaFreeHost(h_HistogramGPU_bins);
	cudaFreeHost(h_HistogramCPU_input);
	cudaFreeHost(h_HistogramCPU_bins);

	return 0;

}