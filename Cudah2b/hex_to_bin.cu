#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <bitset>
#include <string>
#include <algorithm> // For std::remove_if

#define BLOCK_SIZE 256

__global__ void hex_to_bin_kernel(const unsigned char* hex_data, unsigned char* binary_data, size_t num_bytes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we do not exceed the bounds of the hex_data
    if (idx < num_bytes) {
        unsigned char byte1 = hex_data[idx * 2];
        unsigned char byte2 = hex_data[idx * 2 + 1];
        unsigned char value = 0;

        // Convert first hex character to binary
        if (byte1 >= '0' && byte1 <= '9')
            value |= (byte1 - '0') << 4;
        else if (byte1 >= 'a' && byte1 <= 'f')
            value |= (byte1 - 'a' + 10) << 4;
        else if (byte1 >= 'A' && byte1 <= 'F')
            value |= (byte1 - 'A' + 10) << 4;

        // Convert second hex character to binary
        if (byte2 >= '0' && byte2 <= '9')
            value |= (byte2 - '0');
        else if (byte2 >= 'a' && byte2 <= 'f')
            value |= (byte2 - 'a' + 10);
        else if (byte2 >= 'A' && byte2 <= 'F')
            value |= (byte2 - 'A' + 10);

        binary_data[idx] = value; // Store the resulting byte
    }
}

void hex_to_bin_cuda(const std::string& input_path, const std::string& output_path) {
    std::ifstream infile(input_path);
    if (!infile) {
        std::cerr << "Error opening input file: " << input_path << std::endl;
        return;
    }

    std::string hex_data;
    std::string line;

    // Read each line and remove spaces
    while (std::getline(infile, line)) {
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end()); // Remove spaces
        hex_data += line; // Append cleaned line to hex_data
    }
    infile.close();

    // Check for even length of hex data
    if (hex_data.length() % 2 != 0) {
        std::cerr << "Hex data length is odd, invalid input." << std::endl;
        return;
    }

    size_t num_bytes = hex_data.length() / 2; // Number of bytes to process
    std::vector<unsigned char> binary_data(num_bytes);

    unsigned char* d_hex_data = nullptr;
    unsigned char* d_binary_data = nullptr;

    // Allocate GPU memory
    if (cudaMalloc(&d_hex_data, hex_data.length()) != cudaSuccess) {
        std::cerr << "Error allocating GPU memory for d_hex_data." << std::endl;
        return;
    }
    if (cudaMalloc(&d_binary_data, num_bytes) != cudaSuccess) {
        std::cerr << "Error allocating GPU memory for d_binary_data." << std::endl;
        cudaFree(d_hex_data);
        return;
    }

    // Copy data from host to device
    if (cudaMemcpy(d_hex_data, hex_data.data(), hex_data.length(), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Error transferring data to GPU." << std::endl;
        cudaFree(d_hex_data);
        cudaFree(d_binary_data);
        return;
    }

    // Launch the kernel
    hex_to_bin_kernel << <(num_bytes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_hex_data, d_binary_data, num_bytes);

    cudaDeviceSynchronize(); // Wait for the GPU to finish

    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Kernel execution error: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_hex_data);
        cudaFree(d_binary_data);
        return;
    }

    // Copy the result back to host
    if (cudaMemcpy(binary_data.data(), d_binary_data, num_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Error transferring results to host." << std::endl;
        cudaFree(d_hex_data);
        cudaFree(d_binary_data);
        return;
    }

    // Write binary data to output file
    std::ofstream outfile(output_path);
    if (!outfile) {
        std::cerr << "Error opening output file: " << output_path << std::endl;
        cudaFree(d_hex_data);
        cudaFree(d_binary_data);
        return;
    }

    size_t bit_count = 0;
    for (size_t i = 0; i < binary_data.size(); ++i) {
        outfile << std::bitset<8>(binary_data[i]); // Write each byte as 8 bits
        bit_count += 8;

        // After writing 256 bits, insert a new line
        if (bit_count == 256) {
            outfile << std::endl;
            bit_count = 0;
        }
    }

    // If there's leftover binary data not yet newline-separated, add a newline
    if (bit_count > 0) {
        outfile << std::endl;
    }

    outfile.close(); // Close output file

    // Free GPU memory
    cudaFree(d_hex_data);
    cudaFree(d_binary_data);

    std::cout << "Converted " << input_path << " to " << output_path << std::endl;
}
