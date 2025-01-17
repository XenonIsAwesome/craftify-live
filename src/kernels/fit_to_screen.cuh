#ifndef FITTOSCREEN_CUH
#define FITTOSCREEN_CUH

#include <cuda_runtime.h>

namespace craftify {
    namespace kernels {

    __global__ void fit_image_to_screen(
        const uchar4* input_img,
        uchar4* output_img, 
        unsigned int input_width, 
        unsigned int input_height, 
        unsigned int output_width, 
        unsigned int output_height
    );

    }; // namespace kernels
}; // namespace craftify

#endif //FITTOSCREEN_CUH
