cmake_minimum_required(VERSION 3.16)
project(eigen_vis_cpp_cmake_project)

add_subdirectory("extern")

add_executable(eigen_vis_cpp "eigen_vis.cpp")
target_link_libraries(eigen_vis_cpp PRIVATE Eigen)
target_compile_features(eigen_vis_cpp PRIVATE cxx_std_20)
set_target_properties(eigen_vis_cpp PROPERTIES CXX_EXTENSIONS OFF)
