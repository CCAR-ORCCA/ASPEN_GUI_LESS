
cmake_minimum_required(VERSION 3.0.0)

# Building procedure
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(EXE_NAME ${dirName} CACHE STRING "Name of executable to be created.")

project(${EXE_NAME})

# Find Armadillo 
find_package(Armadillo REQUIRED )
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Find Eigen3
find_package(Eigen3 3.1.0 REQUIRED)
include( ${EIGEN3_USE_FILE} )

# Find OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add source files in root directory
add_executable(${EXE_NAME}
main.cpp)


# Linking
set(library_dependencies ${ARMADILLO_LIBRARIES} )


target_link_libraries(${EXE_NAME} ${library_dependencies} OpenMP::OpenMP_CXX)

