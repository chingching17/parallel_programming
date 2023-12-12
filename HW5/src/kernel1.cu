#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel() {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *result_img, *data_img;
    result_img = (int *)malloc(resX * resY * sizeof(int));
    cudaMalloc((void **)&data_img, resX * resY * sizeof(int));

    dim3 ThreadsPerBlock(16, 16);
    dim3 NumOfBlocks(resX / ThreadsPerBlock.x, resY / ThreadsPerBlock.y);
    mandelKernel<<<NumOfBlocks, ThreadsPerBlock>>>(lowerX, lowerY, stepX, stepY, resX, data_img, maxIterations);

}
