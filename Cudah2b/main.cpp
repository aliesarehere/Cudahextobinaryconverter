#include "hex_to_bin.h"
#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include <direct.h>

bool create_directory(const std::string& dir);
void iterate_files(const std::string& dir, std::vector<std::string>& files);

int main() {
    std::string input_dir = "privhexval";
    std::string output_dir = "privbinval";

    if (!create_directory(input_dir) || !create_directory(output_dir)) {
        return 1;
    }

    std::vector<std::string> files;
    iterate_files(input_dir, files);

    if (files.empty()) {
        std::cerr << "Brak plików do przetworzenia w katalogu: " << input_dir << std::endl;
        return 1;
    }

    for (const auto& input_path : files) {
        std::string output_filename = input_path.substr(input_path.find_last_of("/\\") + 1);
        output_filename = output_filename.substr(0, output_filename.find_last_of('.')) + "_bin.txt";
        std::string output_path = output_dir + "\\" + output_filename;

        hex_to_bin_cuda(input_path, output_path);
    }

    return 0;
}

bool create_directory(const std::string& dir) {
    if (_mkdir(dir.c_str()) == 0 || errno == EEXIST) {
        return true;
    }
    else {
        std::cerr << "Nie uda³o siê utworzyæ katalogu: " << dir << std::endl;
        return false;
    }
}

void iterate_files(const std::string& dir, std::vector<std::string>& files) {
    WIN32_FIND_DATA find_file_data;
    HANDLE h_find = FindFirstFile((dir + "\\*").c_str(), &find_file_data);

    if (h_find == INVALID_HANDLE_VALUE) {
        std::cerr << "Nie uda³o siê otworzyæ katalogu: " << dir << std::endl;
        return;
    }

    do {
        const std::string file_name = find_file_data.cFileName;
        const std::string full_file_name = dir + "\\" + file_name;

        if (!(find_file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            files.push_back(full_file_name);
        }
    } while (FindNextFile(h_find, &find_file_data) != 0);

    FindClose(h_find);
}
