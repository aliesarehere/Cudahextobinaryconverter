#ifndef HEX_TO_BIN_H
#define HEX_TO_BIN_H

#include <cuda_runtime.h>
#include <string>

__global__ void hex_to_bin_kernel(const unsigned char* hex_data, unsigned char* binary_data, size_t length);

void hex_to_bin_cuda(const std::string& input_path, const std::string& output_path);

#endif // HEX_TO_BIN_H
