cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS -march=native)
set(CUDA_LIBRARIES /usr/local/cuda-11.6/lib64)
set(CUDA_INCLUDE_DIRS /usr/local/cuda-11.6/include)

#set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr")

#set(TF_INCLUDE_PATH /Users/sunjiahe/miniconda3/envs/tf2.0/lib/python3.6/site-packages/tensorflow_core/include)
set(TF_INCLUDE_PATH /home/vedanshu/.local/lib/python3.10/site-packages/tensorflow/include)

#add_definitions(-D__JETBRAINS_IDE__)
add_definitions(-DEIGEN_USE_THREADS)
option(USE_CUDA OFF)
set(USE_CUDA ON)
if (USE_CUDA)
    add_definitions(-DGOOGLE_CUDA=1)
	add_definitions(-DEIGEN_USE_GPU)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
	set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
	set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr")
endif ()
#add_library(ops SHARED deform_conv_cuda_kernel.h deform_conv_cuda_kernel.cu deform_conv_cuda.cpp)
#add_library(ops SHARED deformable_conv2d.h deformable_conv2d_utils.h test.cpp)
add_library(ops SHARED deformable_conv2d.cpp deformable_conv.cu deformable_conv.cpp deformable_conv2d.h deformable_conv2d_utils.h)
target_include_directories(ops PUBLIC ${CUDA_INCLUDE_DIRS} ${TF_INCLUDE_PATH})
message(STATUS ${TF_INCLUDE_PATH})
#target_link_libraries(ops PRIVATE /Users/sunjiahe/miniconda3/envs/tf2.0/lib/python3.6/site-packages/tensorflow_core/libtensorflow_framework.dylib)
target_link_libraries(ops PUBLIC cuda /home/vedanshu/.local/lib/python3.10/site-packages/tensorflow/libtensorflow_framework.so.2)
#target_compile_options(ops PUBLIC -march=native)
