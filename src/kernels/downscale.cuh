#ifndef DOWNSCALE_CUH
#define DOWNSCALE_CUH

#include <cuda_runtime.h>

namespace craftify {
    namespace kernels {

    __global__ void downscale_image(
        const uchar4* input_img,
        uchar4* output_img, 
        unsigned int input_width, 
        unsigned int input_height, 
        unsigned int output_width, 
        unsigned int output_height, 
        unsigned int scale_factor);

    }; // namespace kernels
}; // namespace craftify

#endif //DOWNSCALE_CUH
