cmake_minimum_required(VERSION 3.22)

project(testApp)

# Point CMake to LibTorch config
set(Torch_DIR "${LibTorch_SOURCE_DIR}/share/cmake/Torch")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(testApp main.cpp
                       ${CMAKE_SOURCE_DIR}/plugin/source/NeuralNetwork.cpp)

target_link_libraries(testApp "${TORCH_LIBRARIES}")
# Set include directories
target_include_directories(testApp PRIVATE "${LibTorch_SOURCE_DIR}/include"
                                           "${CMAKE_SOURCE_DIR}/plugin/source")
