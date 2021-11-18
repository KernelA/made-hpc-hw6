find_package(CUDAToolkit 11.2 REQUIRED)
find_package(openimageio REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)

set(_include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE _headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${_include_dir}/*.h*)


add_executable(${PROJECT_NAME} ${_headers}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/conv_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/conv.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    )

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_include_directories(${PROJECT_NAME} PRIVATE ${_include_dir})
unset(_include_dir)
target_link_libraries(${PROJECT_NAME} PRIVATE opencv_core PRIVATE opencv_imgcodecs PRIVATE CUDA::cudart)
target_precompile_headers(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/stdafx.h)
