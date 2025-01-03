#ifndef DOWNSCALE_CUH
#define DOWNSCALE_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace craftify::kernels {

    __global__ void downscale_image(
        const uchar4* input_img,
        uchar4* output_img, 
        unsigned int input_width, 
        unsigned int input_height, 
        unsigned int output_width, 
        unsigned int output_height, 
        unsigned int scale_factor);

} // craftify_kernels

#endif //DOWNSCALE_CUH
