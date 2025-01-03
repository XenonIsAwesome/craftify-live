#include "downscale.cuh"
#include <ostream>


__global__ void craftify::kernels::downscale_image(
        const uchar4* input_img, 
        uchar4* output_img, 
        unsigned int input_width, 
        unsigned int input_height, 
        unsigned int output_width, 
        unsigned int output_height,
        unsigned int scale_factor
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_width && y < output_height) {
        uint4 sum = make_uint4(0, 0, 0, 0);
        unsigned int count = 0;

        for (int i = 0; i < scale_factor; ++i) {
            for (int j = 0; j < scale_factor; ++j) {
                int input_x = x * scale_factor + i;
                int input_y = y * scale_factor + j;
                if (input_x < input_width && input_y < input_height) {
                    auto pixel = input_img[input_y * input_width + input_x];
                    sum.x += pixel.x;
                    sum.y += pixel.y;
                    sum.z += pixel.z;
                    sum.w += pixel.w;
                    count++;
                }
            }
        }

        (output_img + (y * output_width + x))->x = sum.x / count;
        (output_img + (y * output_width + x))->y = sum.y / count;
        (output_img + (y * output_width + x))->z = sum.z / count;
        (output_img + (y * output_width + x))->w = sum.w / count;
    }
}