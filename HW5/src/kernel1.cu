#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int width, int *data_img, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    float x = lowerX + thisX * stepX;
    float y = lowerY + thisY * stepY;

    float tmpX = x;
    float tmpY = y;
    int idx = thisX + thisY * width;
    for(int i = 0; i < maxIterations; i++){
        if (tmpX * tmpX + tmpY * tmpY > 4.f)break;
        float new_x = tmpX * tmpX - tmpY * tmpY;
        float new_y = 2.f * tmpX * tmpY;
        tmpX = x + new_x;
        tmpY = y + new_y;
    }
    data_img[idx] = i;
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
