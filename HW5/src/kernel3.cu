#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int width, int *img, int maxIterations, int pitch, int PixelsPerThread) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    for(int loop = 0; loop < PixelsPerThread; loop++){
        int thisX = (blockIdx.x * blockDim.x + threadIdx.x) * PixelsPerThread + loop;
        int thisY = blockIdx.y * blockDim.y + threadIdx.y;
        float x = lowerX + thisX * stepX;
        float y = lowerY + thisY * stepY;

        float tmpX = x;
        float tmpY = y;
        int i;
        for (i = 0; i < maxIterations; i++)
        {
            if (tmpX * tmpX + tmpY * tmpY > 4.f)break;
            float new_x = tmpX * tmpX - tmpY * tmpY;
            float new_y = 2.f * tmpX * tmpY;
            tmpX = x + new_x;
            tmpY = y + new_y;
        }
        
        int* row = (int *)((char*)img + thisY * pitch);
        row[thisX] = i;
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *data_img, *h_img;
    size_t pitch;
    int PixelsPerThread = 4;

    cudaHostAlloc((void **)&h_img, resX * resY * sizeof(int), cudaHostAllocDefault);
    cudaMallocPitch((void **)&data_img, &pitch, resX * sizeof(int), resY);

    dim3 ThreadsPerBlock(16, 16);
    dim3 NumOfBlocks(resX/ ThreadsPerBlock.x / PixelsPerThread, resY/ThreadsPerBlock.y);

    mandelKernel<<<NumOfBlocks, ThreadsPerBlock>>>(lowerX, lowerY, stepX, stepY, resX, data_img, maxIterations, pitch, PixelsPerThread);

    cudaMemcpy2D(h_img, resX * sizeof(int), data_img, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, h_img, resX * resY * sizeof(int));

    cudaFree(data_img);
    cudaFreeHost(h_img);

}
// 74.48x