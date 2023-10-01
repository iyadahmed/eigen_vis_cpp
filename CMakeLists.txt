cmake_minimum_required(VERSION 3.16)
project(eigen_vis_cpp_cmake_project)

add_subdirectory("extern")

find_package(OpenMP REQUIRED)

add_executable(eigen_vis_cpp "eigen_vis.cpp")
target_link_libraries(eigen_vis_cpp PRIVATE Eigen OpenMP::OpenMP_CXX)
target_compile_features(eigen_vis_cpp PRIVATE cxx_std_20)
set_target_properties(eigen_vis_cpp PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(eigen_vis_cpp PRIVATE $<$<CONFIG:RelWithDebInfo,Debug>:/fsanitize=address>)
elseif()
    target_compile_options(eigen_vis_cpp PRIVATE $<$<CONFIG:RelWithDebInfo,Debug>:-fsanitize=address -fno-omit-frame-pointer>)
    target_link_options(eigen_vis_cpp PRIVATE $<$<CONFIG:RelWithDebInfo,Debug>-fsanitize=address>)
endif()
