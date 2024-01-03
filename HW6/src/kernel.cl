__kernel void convolution(__global const float *input_img, __global const float *filter,  __global int *imageWidth, __global int *imageHeight, __global int *filterWidth, __global float *output_img) 
{
    int half_filterSize = *filterWidth / 2;
    int x_id = get_global_id(0);
    int y_id = get_global_id(1);
    float sum = 0;
    int j = x_id;
    int i = y_id;
    int k, l;
    for(k = -half_filterSize; k <= half_filterSize; k++){
        for(l = -half_filterSize; l <= half_filterSize; l++){
            if( i + k >= 0 && i + k < *imageHeight && j + l >= 0 && j + l < *imageWidth){
                sum += input_img[(i+k)* *imageWidth + j + l] * filter[(k+half_filterSize) * *filterWidth + l + half_filterSize];
            }
        }
    }
   output_img[i * *imageWidth + j] = sum;
}
