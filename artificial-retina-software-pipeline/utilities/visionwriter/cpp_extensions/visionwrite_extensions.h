#include <string>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py=pybind11;

py::bytes pack_sta_buffer_color (
    py::array_t<float, py::array::c_style | py::array::forcecast>& red_sta,
    py::array_t<float, py::array::c_style | py::array::forcecast>& red_err,
    py::array_t<float, py::array::c_style | py::array::forcecast>& green_sta,
    py::array_t<float, py::array::c_style | py::array::forcecast>& green_err,
    py::array_t<float, py::array::c_style | py::array::forcecast>& blue_sta,
    py::array_t<float, py::array::c_style | py::array::forcecast>& blue_err,
    double refresh_time) {

    /*
    Note that we require that the arrrays have a different array order than the STAs
        returned by visionloader. This is because the original order is a nightmare
        for cache in memory accesses (and hence comically slow for fine
        stimuli)

    We will require an array order swap before using this function
    */

    py::buffer_info red_sta_info = red_sta.request();
    float *red_data_ptr = static_cast<float *> (red_sta_info.ptr);

    size_t sta_depth = red_sta_info.shape[0];
    size_t sta_width = red_sta_info.shape[1];
    size_t sta_height = red_sta_info.shape[2];

    size_t n_output_entries = 6 * sta_width * sta_height * sta_depth + sta_depth * 4;


    py::buffer_info red_err_info = red_err.request();
    float *red_err_ptr = static_cast<float *> (red_err_info.ptr);

    py::buffer_info green_sta_info = green_sta.request();
    float *green_data_ptr = static_cast<float *> (green_sta_info.ptr);

    py::buffer_info green_err_info = green_err.request();
    float *green_err_ptr = static_cast<float *> (green_err_info.ptr);

    py::buffer_info blue_sta_info = blue_sta.request();
    float *blue_data_ptr = static_cast<float *> (blue_sta_info.ptr);

    py::buffer_info blue_err_info = blue_err.request();
    float *blue_err_ptr = static_cast<float *> (blue_err_info.ptr);

    uint32_t *output_buffer = new uint32_t[n_output_entries];

    size_t depth_offset, width_depth_offset, read_offset;
    uint64_t refresh_temp;
    size_t write_idx = 0;
    for (size_t i = 0; i < sta_depth; ++i) {

        depth_offset = i * (sta_width * sta_height);

        output_buffer[write_idx++] = __builtin_bswap32(static_cast<uint32_t>(sta_width));
        output_buffer[write_idx++] = __builtin_bswap32(static_cast<uint32_t>(sta_height));

        refresh_temp = __builtin_bswap64(*(reinterpret_cast<uint64_t *>(&refresh_time)));
        output_buffer[write_idx++] = static_cast<uint32_t> (refresh_temp >> 32);
        output_buffer[write_idx++] = static_cast<uint32_t> (refresh_temp & 0xFFFF);

        for (size_t j = 0; j < sta_width; ++j)  {

            width_depth_offset = j * sta_height + depth_offset;

            for (size_t k = 0; k < sta_height; ++k) {

                read_offset = width_depth_offset + k;

                output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (red_data_ptr + read_offset)));
                output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (red_err_ptr + read_offset)));

                output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (green_data_ptr + read_offset)));
                output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (green_err_ptr + read_offset)));

                output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (blue_data_ptr + read_offset)));
                output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (blue_err_ptr + read_offset)));

            }
        }
    }


    return py::bytes(reinterpret_cast<char *> (output_buffer), n_output_entries * 4);
}
