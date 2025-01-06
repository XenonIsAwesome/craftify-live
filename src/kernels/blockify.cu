#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIDE_SIZE (16)
#define BLOCK_SIZE (BLOCK_SIDE_SIZE * BLOCK_SIDE_SIZE)


namespace craftify {
    namespace kernels {

        __device__ void copy_best_block(
            uchar4* output_img, 
            const uchar4* texture_atlas, 

            unsigned int output_width, 
            unsigned int atlas_width,

            int x, int y,
            int best_block_x, int best_block_y
        ) {
            for (int blocky = 0; blocky < BLOCK_SIDE_SIZE; blocky++) {
                    for (int blockx = 0; blockx < BLOCK_SIDE_SIZE; blockx++) {
                        /// TODO: something fishy going on here, picks blue blocks????
                        auto best_block_pixel_y = best_block_y + blocky;
                        auto best_block_pixel_x = best_block_x + blockx;
                        uchar4 best_block_pixel = texture_atlas[best_block_pixel_y * atlas_width + best_block_pixel_x];

                        auto output_x = (x * BLOCK_SIDE_SIZE + blockx);
                        auto output_y = (y * BLOCK_SIDE_SIZE + blocky);
                        auto output_idx = output_y * output_width + output_x;

                        (output_img + output_idx)->x = best_block_pixel.x;
                        (output_img + output_idx)->y = best_block_pixel.y;
                        (output_img + output_idx)->z = best_block_pixel.z;
                        (output_img + output_idx)->w = best_block_pixel.w;
                    }
                }
        }

        __device__ int calc_deviation(uchar4 a, uchar4 b) {
            int4 result = make_int4(
                abs((int)a.x - (int)b.x),
                abs((int)a.y - (int)b.y),
                abs((int)a.z - (int)b.z),
                abs((int)a.w - (int)b.w)
            );

            return result.x + result.y + result.z;
        }

        __global__ void blockify_image(
            const uchar4* input_img,
            uchar4* output_img, 
            
            const uchar4* texture_atlas,
            const uchar4* avg_colors,
            int avg_colors_size,

            unsigned int input_width, 
            unsigned int input_height, 

            unsigned int output_width, 
            unsigned int output_height,

            unsigned int atlas_width,
            unsigned int atlas_height
        ) {
            unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < input_width && y < input_height) {
                uchar4 input_pixel = input_img[y * input_width + x];

                // TODO: Find a better way to initialize this
                int minimum_deviation = 256 * 256 * 256;
                int best_block_x = 0;
                int best_block_y = 0;
                uchar4 best_avg_color{0,0,0,0};
                bool is_first = true;

                /// Go over all blocks in the atlas to decide the best choice for the pixel
                /// TODO: Optimize this out of the cuda code, using the score system
                unsigned int atlas_grid_height = atlas_height / BLOCK_SIDE_SIZE;
                unsigned int atlas_grid_width = atlas_width / BLOCK_SIDE_SIZE;

                for (int atlas_grid_y = 0; atlas_grid_y < atlas_grid_height; atlas_grid_y++) {
                    for (int atlas_grid_x = 0; atlas_grid_x < atlas_grid_width; atlas_grid_x++) {
                        int color_index = atlas_grid_y * atlas_grid_width + atlas_grid_x;
                        uchar4 avg_color = avg_colors[color_index];

                        if (color_index > avg_colors_size) {
                            continue;
                        }

                        /// Calculate deviation from block average color
                        int current_deviation = calc_deviation(input_pixel, avg_color);

                        if (is_first || current_deviation < minimum_deviation) {
                            minimum_deviation = current_deviation;
                            best_block_x = atlas_grid_x * BLOCK_SIDE_SIZE;
                            best_block_y = atlas_grid_y * BLOCK_SIDE_SIZE;
                            best_avg_color = avg_color;

                            is_first = false;
                        }
                    }
                }

                /// Copy the best block to the output image
                copy_best_block(output_img, texture_atlas, 
                    output_width, atlas_width, 
                    x, y ,best_block_x, best_block_y);
            }
        }

    }; // namespace kernels
}; // namespace craftify