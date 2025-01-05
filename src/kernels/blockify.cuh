#ifndef BLOCKIFY_CUH
#define BLOCKIFY_CUH

#include <cuda_runtime.h>

namespace craftify {
    namespace kernels {

    __device__ int calc_deviation(uchar4 a, uchar4 b);

    __global__ void blockify_image(
        const uchar4 *input_img,
        uchar4 *output_img, 

        const uchar4 *texture_atlas,
        const uchar4 *avg_colors,
        int avg_colors_length,

        unsigned int input_width, 
        unsigned int input_height, 

        unsigned int output_width, 
        unsigned int output_height,

        unsigned int atlas_width,
        unsigned int atlas_height);

    }; // namespace kernels
}; // namespace craftify

#endif //BLOCKIFY_CUH
