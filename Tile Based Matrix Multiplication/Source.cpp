
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <device_functions.h>

#include <stdio.h>
#include <cstdlib>
#include <cassert>
#include <iostream>

using std::cout;
using std::cin;
using std::endl;


constexpr auto tileWidth = 4;


/*


./TiledMatrixMul -i <rowDimA>(m) <colDimA>(n) <colDimB>(k)
    A = m x n
    B = n x k
    c = m x k

*/
//#define The_index_of_the_refrenced_thread = threadIdx



//initialize square matrix with random numbers
void init_matrix(float* m, int N) { // pass in size of matrix A, 
    for (int i = 0; i < N; i++) {
        m[i] = rand() % 10;
    }
}

void printMatrix(float* m, int n)
{
    for (int i = 0; i < n; i++)
        std::cout << m[i] << " ";
}





__global__ void matrixMul(float* A, float* B, float* C, int numRowsA, int numColA, int numRowsB, int numColB, int numRowsC, int numColC) {


    //allocate shared memory
    __shared__ int sharedA[tileWidth][tileWidth];
    __shared__ int sharedB[tileWidth][tileWidth];

    //extract some built in values to simplify code
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    //calculate each threads global row and column
    int row = by * tileWidth + ty;
    int col = bx * tileWidth + tx;

    float cValue = 0.0;

    if (numColA != numRowsB) return;
    for (int i = 0; i < (int)(ceil((float)numColA / tileWidth)); i++)
    {

        if (i * tileWidth + tx < numColA && row < numRowsA) {
            sharedA[ty][tx] = A[row * numColA + i * tileWidth + tx];
        }
        else {
            sharedA[ty][tx] = 0.0;
        }

        if (i * tileWidth + ty < numRowsB && col < numColB) {
            sharedB[ty][tx] = B[(i * tileWidth + ty) * numColB + col];
        }
        else {
            sharedB[ty][tx] = 0.0;
        }
        __syncthreads();
        if (row < numRowsA && col < numColB) {

            for (int j = 0; j < tileWidth; j++) {
                cValue += sharedA[ty][j] * sharedB[j][tx];
            }
        }

        __syncthreads();
    }

    if (row < numRowsC && col < numColC)
        C[row * numColC + col] = cValue;


}


void matrixMulCPU(float* A, float* B, float* C,
    int numRowsA, int numColA, int numRowsB, int numColB, int numRowsC, int numColC) {

    float temp;
    //loop for each element in the row, for each row, for each column 
    for (int i = 0; i < numRowsA; i++) {
        for (int j = 0; j < numColB; j++) {
            temp = 0;
            for (int k = 0; k < numColA; k++) {
                temp += A[i * numColA + k] * B[k * numColB + j];
            }
            assert(temp == C[i * numColB + j]);
            //cout << temp << " ";

        }

    }

}






int main(int argc, char** argv) {
    //matrix dim default  and blocksize default
    int block_size = 32;
    dim3 dimsA(10, 10);
    dim3 dimsB(10, 10);

    if (argc == 4)
    {
        assert(atoi(argv[1]) <= 0);
        dimsA.y = atoi(argv[1]);

        assert(atoi(argv[2]) <= 0);
        dimsA.x = atoi(argv[2]);

        assert(atoi(argv[3]) <= 0);
        dimsB.x = atoi(argv[3]);
    }


    dimsB.y = dimsA.x;
    dim3 dimsC(dimsB.x, dimsA.y, 1);



    //===============================================

    //big O of for loop in kernal times threadcount
    //======================================


    //allocate memory for host
    float* h_A, * h_B, * h_C;

    //allocate memory for host matrix A
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    cudaMallocHost((void**)&h_A, mem_size_A);

    //allocate memory for host matrix B
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    cudaMallocHost((void**)&h_B, mem_size_B);

    //allocate memory for host matrix C
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    cudaMallocHost((void**)&h_C, mem_size_C);

    cudaStream_t stream;

    //inital matrix dimensions and values
    int initA = dimsA.x * dimsA.y;
    int initB = dimsB.x * dimsB.y;
    init_matrix(h_A, initA);
    init_matrix(h_B, initB);

    //printMatrix(h_A, initA);
    //cout << endl; 
    //printMatrix(h_B, initB);





    //initialize memory for device
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, mem_size_A);
    cudaMalloc(&d_B, mem_size_B);
    cudaMalloc(&d_C, mem_size_C);

    //create cuda event start/stop
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);


    //copy host mem to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    //setup execution parameters / instantiate kernal grid dimensions with padding
    dim3 threads(tileWidth, tileWidth);
    dim3 grid((dimsB.x + threads.x - 1) / threads.x, (dimsA.y + threads.y - 1) / threads.y); //grid = colB/32 , 

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Perform warmup operation using matrixMul CUDA kernel
    matrixMul << <grid, threads >> > (d_A, d_B, d_C, dimsA.y, dimsA.x, dimsB.y, dimsB.x, dimsC.y, dimsC.x);

    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);

    //launch the kernal
    int nIter = 300;
    for (int j = 0; j < nIter; j++) {
        matrixMul << <grid, threads >> > (d_A, d_B, d_C, dimsA.y, dimsA.x, dimsB.y, dimsB.x, dimsC.y, dimsC.x);
    }

    //stop the recording of cuda event
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    //compute performance in seconds 
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    // Compute and print the performance
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

    //cout << "milliseconds per matrix multiplication: " << msecPerMatrixMul << endl;

    //copy results back into host
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    //printMatrix(h_A, size_A);
    //cout << "              " << endl;
    //printMatrix(h_B, size_B);


    int printNum = dimsB.x * dimsA.y;
    //printMatrix(h_C, printNum);

    matrixMulCPU(h_A, h_B, h_C, dimsA.y, dimsA.x, dimsB.y, dimsB.x, dimsC.y, dimsC.x);
    cout << "result verified true" << endl;






    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}





