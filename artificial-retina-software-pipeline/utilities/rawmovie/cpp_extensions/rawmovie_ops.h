#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdint.h>

#if defined(ENABLE_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0; }
inline omp_int_t omp_get_max_threads() { return 1; }
inline void omp_set_num_threads(int num_threads) { return; }
#endif

#define N_COLOR_CHANS 3

namespace py=pybind11;

template<class T>
using ContigNPArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

static const float RGB_CONVERSION_FLOAT[] = {0.2989f, 0.5870f, 0.1140f};
static const double RGB_CONVERSION_DOUBLE[] = {0.2989, 0.5870, 0.1140};


ContigNPArray<float> convert_color_to_bw_float32(
        ContigNPArray<uint8_t> color_three_chan) {

    py::buffer_info color_im_info = color_three_chan.request();
    int64_t n_frames = color_im_info.shape[0];
    int64_t height = color_im_info.shape[1];
    int64_t width = color_im_info.shape[2];
    int64_t three = color_im_info.shape[3];

    uint8_t *color_base_ptr = static_cast<uint8_t *>(color_im_info.ptr);

    auto output_buffer_info = py::buffer_info(
        nullptr,
        sizeof(float),
        py::format_descriptor<float>::value,
        3,
        {n_frames, height, width},
        {sizeof(float) * height * width, sizeof(float) * width, sizeof(float)}
    );

    ContigNPArray<float> bw_output = ContigNPArray<float>(output_buffer_info);
    py::buffer_info output_info = bw_output.request();
    float *output_base_ptr = static_cast<float *> (output_info.ptr);

    omp_set_num_threads(8);
    #pragma omp parallel for
    for (int64_t i = 0; i < n_frames; ++i) {
	int64_t frame_offset = i * height * width;
        for (int64_t j = 0; j < height; ++j) {
	    int64_t base_offset = frame_offset + (j * width);
	    int64_t read_offset, write_offset;
            for (int64_t k = 0; k < width; ++k) {

		write_offset = base_offset + k;
		read_offset = N_COLOR_CHANS * write_offset;

                float acc = 0.0f;
                for (int64_t m = 0; m < 3; ++m) {
                    acc += color_base_ptr[read_offset + m] * RGB_CONVERSION_FLOAT[m];
                }

		*(output_base_ptr + write_offset) = acc;
            }
        }
    }

    return bw_output;
}


