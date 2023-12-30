#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int img_size = imageHeight * imageWidth;
    cl_int ret;

    // create command queue
    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, &ret);

    // create kernel memory
    cl_mem input_img_memory = clCreateBuffer(*context, CL_MEM_READ_WRITE,  sizeof(float) * img_size, NULL, &ret);
    cl_mem filter_memory = clCreateBuffer(*context, CL_MEM_READ_WRITE, sizeof(float) * filterSize, NULL, &ret);
    cl_mem Height = clCreateBuffer(*context, CL_MEM_READ_WRITE, sizeof(int), NULL, &ret);
    cl_mem Width = clCreateBuffer(*context, CL_MEM_READ_WRITE, sizeof(int), NULL, &ret);
    cl_mem filterWidth_memory = clCreateBuffer(*context, CL_MEM_READ_WRITE, sizeof(int), NULL, &ret);
    cl_mem output_image_memory = clCreateBuffer(*context, CL_MEM_READ_WRITE, sizeof(float) * img_size, NULL, &ret);

    // init memory for memory copy
    ret = clEnqueueWriteBuffer(command_queue, input_img_memory, CL_TRUE, 0, img_size * sizeof(float), inputImage, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, filter_memory, CL_TRUE, 0, filterSize * sizeof(float), filter, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, Height, CL_TRUE, 0, sizeof(int), &imageHeight, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, Width, CL_TRUE, 0, sizeof(int), &imageWidth, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, filterWidth_memory, CL_TRUE, 0, sizeof(int), &filterWidth, 0, NULL, NULL);

    // create kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &ret);

    // set kernel arg
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_img_memory);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter_memory);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&Width);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&Height);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&filterWidth_memory);
    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&output_image_memory);

    // run kernel
    size_t global_size[2] = {imageWidth, imageHeight};
    size_t local_size[2] = {10, 10};
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, &global_size, &local_size, 0, NULL, NULL);

    // copy result to host
    ret = clEnqueueReadBuffer(command_queue, output_image_memory, CL_TRUE, 0, img_size * sizeof(float), outputImage, 0, NULL, NULL);

    // free memory
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);

    ret = clReleaseMemObject(input_img_memory);
    ret = clReleaseMemObject(filter_memory);
    ret = clReleaseMemObject(Width);
    ret = clReleaseMemObject(Height);
    ret = clReleaseMemObject(filterWidth_memory);
    ret = clReleaseMemObject(output_image_memory);

    ret = clReleaseCommandQueue(command_queue);

}