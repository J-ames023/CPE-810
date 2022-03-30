//HW1 submission for James Fong
//Corraborators with Kerim Karabacak and Matthew Lepis, massive help from them as I did not understand a lot of what to ask
//All sources are either listed on this code, derived from the practice example in CUDA for Matrix Multiplication, or listed in the HW1 Lab Report.

#include <cuda_runtime.h> 
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h> 

using std::cout;
using std::cin;
using std::endl;

constexpr auto TILE_WIDTH = 16;

// see implimentation similarly used by Matt with assistance from on stackoverflow: https://stackoverflow.com/questions/13896560/multiply-rectangular-matrices-in-cuda

__global__ void CUDAMatrixMultiply(float* a, float* b, float* c, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
    //allocating shared memory
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    //extract some built in values to simplified
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0;

    if (numAColumns != numBRows) return;
    for (int i = 0; i < (int)(ceil((float)numAColumns / TILE_WIDTH)); i++)
    {

        if (i * TILE_WIDTH + tx < numAColumns && Row < numARows) {
            sharedA[ty][tx] = a[Row * numAColumns + i * TILE_WIDTH + tx];
        }
        else {
            sharedA[ty][tx] = 0.0;
        }

        if (i * TILE_WIDTH + ty < numAColumns && Col < numAColumns) {
            sharedB[ty][tx] = b[(i * TILE_WIDTH + ty) * numBColumns + Col];
        }
        else {
            sharedB[ty][tx] = 0.0;
        }

        __syncthreads();

        if (Row < numARows && Col < numBColumns) {
            for (int j = 0; j < TILE_WIDTH; j++) {
                Cvalue += sharedA[ty][j] * sharedB[j][tx];
            }
        }

        __syncthreads();

    }

    if (Row < numCRows && Col < numCColumns) {
        c[Row * numCColumns + Col] = Cvalue;
    }
}

void verify_result(float* a, float* b, float* c,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns)
{
    float tmp;

    //For Every Row
    for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numBColumns; j++) {
            tmp = 0;
            for (int k = 0; k < numAColumns; k++) {
                tmp += a[i * numAColumns + k] * b[k * numBColumns + j];
            }
            assert(tmp == c[i * numBColumns + j]);
        }
    }
}


// initializing matrix with random numbers
void initializeMatrix(float* m, int n) {
    for (int i = 0; i < n; i++)
        m[i] = rand() % 10;
}

//prints out matrix
void printMatrix(float* m, int n) {
    for (int i = 0; i < n; i++)
        cout << m[i] << " ";
}

int main(int argc, char** argv) {
    dim3 dimsA;
    dim3 dimsB;
    dim3 dimsC;

    if (argc == 4) {
        assert(atoi(argv[1]) <= 0);
        dimsA.y = atoi(argv[1]);

        assert(atoi(argv[2]) <= 0);
        dimsA.x = atoi(argv[2]);

        assert(atoi(argv[3]) <= 0);
        dimsB.x = atoi(argv[3]);

        cout << "Command Lines accepted" << endl;
    }

    else {

        cout << "Enter row dimensions for matrix A: " << endl;
        cin >> dimsA.y;

        cout << "Enter column dimensions for matrix A: " << endl;
        cin >> dimsA.x;

        cout << "Enter row dimensions for matrix B: " << endl;
        cin >> dimsB.x;
    }

    dimsB.y = dimsA.x;
    dimsC.y = dimsA.y;
    dimsC.x = dimsB.x;

    //getting bytes size requirement for memory allocation
    
    unsigned int Size_A = (dimsA.y * dimsA.x);
    unsigned int Size_B = (dimsB.y * dimsB.x);
    unsigned int Size_C = (dimsC.y * dimsC.x);

    unsigned int mem_size_A = Size_A * sizeof(float);
    unsigned int mem_size_B = Size_B * sizeof(float);
    unsigned int mem_size_C = Size_C * sizeof(float);

    //Allocating HOST memory for the matrices.
    float* h_A, * h_B, * h_C;

    //Matrix A
    cudaMallocHost((void**)&h_A, mem_size_A);
    checkCudaErrors(cudaMallocHost((void**)&h_A, mem_size_A));

    //Matrix B
    cudaMallocHost((void**)&h_B, mem_size_B);
    checkCudaErrors(cudaMallocHost((void**)&h_B, mem_size_B));

    //Matrix C
    cudaMallocHost((void**)&h_C, mem_size_C);
    checkCudaErrors(cudaMallocHost((void**)&h_C, mem_size_C));

    // initialize the matrixes 
    initializeMatrix(h_A, Size_A);
    initializeMatrix(h_B, Size_B);

    //Allocating DEVICE memory for the matrices.
    float* a_D, * b_D, * c_D;
    cudaMalloc(&a_D, mem_size_A);
    cudaMalloc(&b_D, mem_size_B);
    cudaMalloc(&c_D, mem_size_C);

    //create cuda event start/stop
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStream_t stream;

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    //copying matrices from HOST memory to DEVICE memory.
    cudaMemcpy(a_D, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(b_D, h_B, mem_size_B, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    // Establishing grid and block dimensions.
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((dimsB.x + threads.x - 1) / threads.x, (dimsA.y + threads.y - 1) / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Perform warmup operation using matrixMul CUDA kernel
    // KERNEL CALL 
    CUDAMatrixMultiply <<<grid, threads>>> (a_D, b_D, c_D, dimsA.y, dimsA.x, dimsB.y, dimsB.x, dimsC.y, dimsC.x);
    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);

    //Copy Results from Device to Host
    cudaMemcpy(h_C, c_D, mem_size_C, cudaMemcpyDeviceToHost);

    //launch the kernel
    int nIter = 300;
    for (int j = 0; j < nIter; j++) {
        CUDAMatrixMultiply <<<grid, threads >>> (a_D, b_D, c_D, dimsA.y, dimsA.x, dimsB.y, dimsB.x, dimsC.y, dimsC.x);
    }

    //stop the recording of cuda event
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    verify_result(h_A, h_B, h_C, dimsA.y, dimsA.x, dimsB.y, dimsB.x, dimsC.y, dimsC.x);
    cout << "SUCCESSFUL EXECUTION." << endl;

    // Compute and print the performance 
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
        static_cast<double>(dimsA.y) *
        static_cast<double>(dimsB.x);
    double gigaFlops =
        (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
        " WorkgroupSize= %u threads/block\n",
        gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

    //Results return to HOST
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, c_D, mem_size_C, cudaMemcpyDeviceToHost);

    /*
    printMatrix(h_A, Size_A);
    cout << endl;
    printMatrix(h_B, Size_B);
    cout << endl;

    printMatrix(h_C, Size_C);
    cout << endl;
    */

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(a_D);
    cudaFree(b_D);
    cudaFree(c_D);

    return 0;
}