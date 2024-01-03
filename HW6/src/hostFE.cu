#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "helper.h"
extern "C"{
#include "hostFE.h"
}

__global__ void convolution(int filterWidth, float *filter, int imageHeight, int imageWidth, float *inputImage, float *outputImage){
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = thisX + thisY * imageWidth;

    int half_filterSize = filterWidth / 2;
    float sum = 0;
    int k, l; 
    for(k = -half_filterSize; k <= half_filterSize; k++){
        for(l = -half_filterSize; l <= half_filterSize; l++){
            if(thisY + k >= 0 && thisY + k < imageHeight && thisX + l >= 0 && thisX + l < imageWidth){
                sum += inputImage[(thisY + k) * imageWidth + thisX + l] * filter[(k + half_filterSize) * filterWidth + l + half_filterSize];
            }

        }
    }
    outputImage[idx] = sum;
}

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    // set size
    int filterSize = filterWidth * filterWidth;
    int mem_size = imageHeight * imageWidth;

    // allocate memory
    float *data_filter, *data_inputImage, *data_outputImage;
    cudaMalloc(&data_filter, filterSize * sizeof(float));
    cudaMalloc(&data_inputImage, mem_size * sizeof(float));
    cudaMalloc(&data_outputImage, mem_size * sizeof(float));

    // copy host to cuda device
    cudaMemcpy(data_filter, filter, filterSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_inputImage, inputImage, mem_size * sizeof(float), cudaMemcpyHostToDevice);

    // run CUDA
    dim3 threadsPerBlock(25, 25);
    dim3 numBlocks(imageWidth / threadsPerBlock.x, imageHeight / threadsPerBlock. y);
    convolution<<<numBlocks, threadsPerBlock>>>(filterWidth, data_filter, imageHeight, imageWidth, data_inputImage, data_outputImage);

    // copy result to host
    cudaMemcpy(outputImage, data_outputImage, mem_size * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(data_filter);
    cudaFree(data_inputImage);
    cudaFree(data_outputImage);
}