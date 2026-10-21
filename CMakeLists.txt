cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(nbody LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Sin optimización la comparación con la CPU no sería justa
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(nbody INTERFACE)
target_include_directories(nbody INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(demo test/demo.cu)
target_link_libraries(demo PRIVATE nbody)