//HW3 submission for James Fong.
//Corraborators with Kerim Karabacak and Matthew Lepis.
//All sources are either listed on this code, derived from the practice example in CUDA for convolution using textured and shared memory.


#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <device_functions.h>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <helper_cuda.h>
#include <helper_functions.h> 

using namespace std;

//Maps to a single instruction on G8x / G9x / G10x, GPU-specific defines
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b)
{
    return (a % b != 0) ? (a - a % b + b) : a;
}

// CONVOLUTIONTEXTURE_gold beginning

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowsCPU(
    float* h_Dst,
    float* h_Src,
    float* h_Kernel,
    int imageW,
    int imageH,
    int kernelR
)
{
    for (int y = 0; y < imageH; y++)
        for (int x = 0; x < imageW; x++)
        {
            float sum = 0;

            for (int k = -kernelR; k <= kernelR; k++)
            {
                int d = x + k;

                if (d < 0) d = 0;

                if (d >= imageW) d = imageW - 1;

                sum += h_Src[y * imageW + d] * h_Kernel[kernelR - k];
            }

            h_Dst[y * imageW + x] = sum;
        }
}
//void convolutionRowsCPU


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnsCPU(
    float* h_Dst,
    float* h_Src,
    float* h_Kernel,
    int imageW,
    int imageH,
    int kernelR
)
{
    for (int y = 0; y < imageH; y++)
        for (int x = 0; x < imageW; x++)
        {
            float sum = 0;

            for (int k = -kernelR; k <= kernelR; k++)
            {
                int d = y + k;

                if (d < 0) d = 0;

                if (d >= imageH) d = imageH - 1;

                sum += h_Src[d * imageW + x] * h_Kernel[kernelR - k];
            }

            h_Dst[y * imageW + x] = sum;
        }
}

//__global__ void ConvolutionRowsGPUKernal

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowsKernel(
    float* d_Dst,
    int imageW,
    int imageH,
    float* d_Mask,
    int KERNEL_RADIUS,
    cudaTextureObject_t texSrc
)
{
    //allocate shared memory
    extern __shared__ float c_Kernel[];

    //initialize shared memory
    if (threadIdx.x < (2 * KERNEL_RADIUS)) {
        c_Kernel[threadIdx.x] = d_Mask[threadIdx.x];
    }

    //wait for threads to clear shared memory
    __syncthreads();

    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if (ix >= imageW || iy >= imageH)
    {
        return;
    }

    float sum = 0;

    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
    {
        sum += tex2D<float>(texSrc, x + (float)k, y) * c_Kernel[KERNEL_RADIUS - k];
    }

    d_Dst[IMAD(iy, imageW, ix)] = sum;
}

//__global__ void ConvolutionRowsGPUKernal

//start of void convolutionRowsGPU

void convolutionRowsGPU(float* d_Dst, int imageW, int imageH, int threadCount, float* d_Mask, int KERNEL_RADIUS, cudaTextureObject_t texSrc, size_t SHARED_MEM) {
    int elements = imageW * imageH;
    int THREADS = threadCount * threadCount;
    int BLOCKS = (elements + THREADS - 1) / THREADS;

    convolutionRowsKernel << <BLOCKS, THREADS, SHARED_MEM >> > (d_Dst, imageW, imageH, d_Mask, KERNEL_RADIUS, texSrc);
}

//end of void convolutionRowsGPU

//start of __global__ void ConvolutioncolsGPUKernal
////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////

__global__ void convolutionColumnsGPUKernel(
    float* d_Dst,
    int imageW,
    int imageH,
    float* d_Mask,
    int KERNEL_RADIUS,
    cudaTextureObject_t texSrc
)
{
    //allocate shared memory
    extern __shared__ float c_Kernel[];

    //initialize shared memory
    if (threadIdx.x < (2 * KERNEL_RADIUS)) {
        c_Kernel[threadIdx.x] = d_Mask[threadIdx.x];
    }

    //wait for threads to clear shared memory
    __syncthreads();

    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if (ix >= imageW || iy >= imageH)
    {
        return;
    }

    float sum = 0;

    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
    {
        sum += tex2D<float>(texSrc, x, y + (float)k) * c_Kernel[KERNEL_RADIUS - k];
    }

    d_Dst[IMAD(iy, imageW, ix)] = sum;
}

//end of __global__ void ConvolutioncolsGPUKernal

//start of void convolutionColsGPU

void convolutionColumnsGPU(float* d_Dst, int imageW, int imageH, int threadCount, float* d_Mask, int KERNEL_RADIUS, cudaTextureObject_t texSrc, size_t SHARED_MEM) {
    int elements = imageW * imageH;
    int THREADS = threadCount * threadCount;
    int BLOCKS = (elements + THREADS - 1) / THREADS;


    convolutionColumnsGPUKernel << <BLOCKS, THREADS, SHARED_MEM >> > (d_Dst, imageW, imageH, d_Mask, KERNEL_RADIUS, texSrc);
}
//End of void convolutionColsGPU

//End of convolutionTexture.cu


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    float //inital variable declarations
        * h_Kernel,
        * h_Input,
        * h_Buffer,
        * h_OutputCPU,
        * h_OutputGPU,
        * d_Output,
        gpuTime;

    cudaArray* a_Src;
    cudaTextureObject_t texSrc;
    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();

    StopWatchInterface* hTimer = NULL;
    sdkCreateTimer(&hTimer);

    dim3 dimsInputMatrix;   //Dimensions for Input Matrix
    int maskLength;         //Mask Length
    int threadCount;  //Thread Count
    int iterations;

    if (argc == 4) {
        if (atoi(argv[1]) > 0)
            dimsInputMatrix.y = atoi(argv[1]); //Row count for Input Matrix of image

        if (atoi(argv[2]) > 0)
            dimsInputMatrix.x = atoi(argv[2]); //Columns count for Input Matrix of image

        dimsInputMatrix.z = dimsInputMatrix.x * dimsInputMatrix.y;  //Total number of elements in the input matrix

        if (atoi(argv[3]) > 0)
            maskLength = atoi(argv[3]); //Argument for Mask Length
        else {
            cout << "Command Line Argument not accepted." << endl;
            return 0;
        }

        cout << "Three Command Line Arguments Accepted." << endl;
    }

    else if (argc == 5) {
        if (atoi(argv[1]) > 0)
            dimsInputMatrix.y = atoi(argv[1]);

        if (atoi(argv[2]) > 0)
            dimsInputMatrix.x = atoi(argv[2]);

        dimsInputMatrix.z = dimsInputMatrix.x * dimsInputMatrix.y; //Total number of elements in the input matrix

        if (atoi(argv[3]) > 0)
            maskLength = atoi(argv[3]);

        if (atoi(argv[4]) > 0)
            threadCount = atoi(argv[4]); // Argument for thread length
        else {
            cout << "Command Line Argument not accepted." << endl;
            return 0;
        }

        cout << "Four Command Line Arguments Accepted." << endl;
    }

    else if (argc == 6) {
        if (atoi(argv[1]) > 0)
            dimsInputMatrix.y = atoi(argv[1]);

        if (atoi(argv[2]) > 0)
            dimsInputMatrix.x = atoi(argv[2]);

        dimsInputMatrix.z = dimsInputMatrix.x * dimsInputMatrix.y; //Total number of elements in the input matrix

        if (atoi(argv[3]) > 0)
            maskLength = atoi(argv[3]);

        if (atoi(argv[4]) > 0)
            threadCount = atoi(argv[4]);

        if (atoi(argv[5]) > 0)
            iterations = atoi(argv[5]); //Total number of iterations.
        else {
            cout << "Command Line Argument not accepted." << endl;
            return 0;
        }

        cout << "Five Command Line Arguments Accepted." << endl;
    }

    else {
        cout << "Enter row dimensions: ";
        cin >> dimsInputMatrix.y;

        cout << "Enter column dimensions: ";
        cin >> dimsInputMatrix.x;

        //Total number of elements in the input matrix
        dimsInputMatrix.z = dimsInputMatrix.x * dimsInputMatrix.y;

        cout << "Enter the length of the mask: ";
        cin >> maskLength;

        cout << "Enter the threads per block: ";
        cin >> threadCount;

        if (threadCount > 32)
        {
            cout << "Maximum number of threads is 32, thus setting current number of threads as 32" << endl;
            threadCount = 32;
        }

        cout << "Enter the iterations per block: ";
        cin >> iterations;

    }

    const int imageW = dimsInputMatrix.y; // Size of image
    const int imageH = dimsInputMatrix.x;

    printf("[%s] - Starting...\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    findCudaDevice(argc, (const char**)argv);

    printf("Initializing data...\n");
    h_Kernel = (float*)malloc(maskLength * sizeof(float));
    h_Input = (float*)malloc(imageW * imageH * sizeof(float));
    h_Buffer = (float*)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float*)malloc(imageW * imageH * sizeof(float));
    h_OutputGPU = (float*)malloc(imageW * imageH * sizeof(float));

    cudaMallocArray(&a_Src, &floatTex, imageW, imageH);
    cudaMalloc((void**)&d_Output, imageW * imageH * sizeof(float));

    cudaResourceDesc            texRes;             // Resource Descriptor
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;   // Assigning resource type of texRes to be a cuda Array cudaResourceTypeArray
    texRes.res.array.array = a_Src;                   // source for the new point in the array being constructed.

    cudaTextureDesc             texDescr;                // Texture Descriptor
    memset(&texDescr, 0, sizeof(cudaTextureDesc));         // Sets memory for texture descriptor

    texDescr.normalizedCoords = false;                   // ensures texture reads were not normalized
    texDescr.filterMode = cudaFilterModeLinear;    // Linear convolution
    texDescr.addressMode[0] = cudaAddressModeWrap;       // The signal c[k / M] is continued outside k=0, ..., M-1 so that it is periodic with period equal to M. In other words, c[(k + p * M) / M] = c[k / M] for any (positive, negative or vanishing) integer p.
    texDescr.addressMode[1] = cudaAddressModeWrap;       // https://stackoverflow.com/questions/19020963/the-different-addressing-modes-of-cuda-textures
    texDescr.readMode = cudaReadModeElementType;         // Reads elements inserted

    cudaCreateTextureObject(&texSrc, &texRes, &texDescr, NULL);    //Create Texture Object

    srand(2009);        // Date that the code was written

    for (unsigned int i = 0; i < maskLength; i++)
    {
        h_Kernel[i] = (float)(rand() % 16);         //Random variable mask array from integers 1 to 15
    }

    for (unsigned int i = 0; i < imageW * imageH; i++)
    {
        h_Input[i] = (float)(rand() % 16);          //Random variable input from integers 1 to 15
    }

    //setConvolutionKernel(h_Kernel);                 //Set Convolution Mask through GPU texture-based convolution
    cudaMemcpyToArray(a_Src, 0, 0, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice);          //


    printf("Running GPU rows convolution (%u identical iterations)...\n", iterations);
    cudaDeviceSynchronize();
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    int KERNEL_RADIUS = maskLength / 2;
    size_t shared_mem = dimsInputMatrix.z * sizeof(float);

    for (unsigned int i = 0; i < iterations; i++) // Run Convolution Rows GPU with identical iteration
    {
        convolutionRowsGPU(
            d_Output,
            imageW,
            imageH,
            threadCount,
            h_Kernel,
            KERNEL_RADIUS,
            texSrc,
            shared_mem
        );
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
    printf("Average convolutionRowsGPU() time: %f msecs; //%f Mpix/s\n", gpuTime, imageW * imageH * 1e-6 / (0.001 * gpuTime));

    //While CUDA kernels can't write to textures directly, this copy is inevitable
    printf("Copying convolutionRowGPU() output back to the texture...\n");
    cudaDeviceSynchronize();
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    cudaMemcpyToArray(a_Src, 0, 0, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    printf("cudaMemcpyToArray() time: %f msecs; //%f Mpix/s\n", gpuTime, imageW * imageH * 1e-6 / (0.001 * gpuTime));

    printf("Running GPU columns convolution (%i iterations)\n", iterations);
    cudaDeviceSynchronize();
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (int i = 0; i < iterations; i++)
    {
        convolutionColumnsGPU(
            d_Output,
            imageW,
            imageH,
            threadCount,
            h_Kernel,
            KERNEL_RADIUS,
            texSrc,
            shared_mem
        );
    }

    // sync, calculate and display calculation time for Col Convolution on GPU
    cudaDeviceSynchronize();
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
    printf("Average convolutionColumnsGPU() time: %f msecs; //%f Mpix/s\n", gpuTime, imageW * imageH * 1e-6 / (0.001 * gpuTime));

    // Copy final results back to host
    printf("Reading back GPU results...\n");
    cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost);

    //run convolution on CPU to test GPU calculations, first rows then columns
    printf("Checking the results...\n");
    printf("...running convolutionRowsCPU()\n");
    convolutionRowsCPU(
        h_Buffer,
        h_Input,
        h_Kernel,
        imageW,
        imageH,
        KERNEL_RADIUS
    );

    printf("...running convolutionColumnsCPU()\n");
    convolutionColumnsCPU(
        h_OutputCPU,
        h_Buffer,
        h_Kernel,
        imageW,
        imageH,
        KERNEL_RADIUS
    );

    double delta = 0;
    double sum = 0;

    for (unsigned int i = 0; i < imageW * imageH; i++)
    {
        sum += h_OutputCPU[i] * h_OutputCPU[i];
        delta += (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
    }

    double L2norm = sqrt(delta / sum);
    printf("Relative L2 norm: %E\n", L2norm);
    printf("Shutting down...\n");

    cudaFree(d_Output);
    cudaFreeArray(a_Src);
    free(h_OutputGPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Kernel);

    sdkDeleteTimer(&hTimer);

    if (L2norm > 1e-6)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
