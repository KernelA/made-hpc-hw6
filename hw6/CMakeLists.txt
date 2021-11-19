find_package(CUDAToolkit 11.2 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)

set(_include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE _headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${_include_dir}/*.h*)


SET(test_projects "conv;median_filter")

foreach(test_project ${test_projects})
    add_executable(${test_project} ${_headers}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/${test_project}.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/conv_kernel.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/conv.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/src/median_filter.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/median_filter_kernel.cu
    )

    target_compile_features(${test_project} PRIVATE cxx_std_17)
    target_include_directories(${test_project} PRIVATE ${_include_dir})
    target_link_libraries(${test_project} PRIVATE opencv_core PRIVATE opencv_imgcodecs PRIVATE CUDA::cudart)
    target_precompile_headers(${test_project} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/stdafx.h)
endforeach()


