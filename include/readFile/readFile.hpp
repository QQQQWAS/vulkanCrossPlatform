#pragma once

#include <ios>
#include <iostream>
#include <string>

namespace rf{
  std::string readFile(const char* path, std::ios_base::openmode openmode = std::iostream::in);
}
