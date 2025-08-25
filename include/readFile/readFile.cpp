#include "readFile.hpp"

#include <fstream>
#include <string>

namespace rf{
  std::string readFile(const char* path, std::ios_base::openmode openmode){
    std::ifstream file(path, openmode);
    if(!file){
      printf("%s is not a valid file path\n", path);
      return "";
    }
    std::string line, text;
    while (std::getline(file, line)) {
      text.append(line + "\n");
    }
    file.close();
    return text;
  }
}
