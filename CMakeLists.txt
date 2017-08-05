# CMakeLists.txt for ASPEN_GUI_LESS
# Benjamin Bercovici, 03/10/2017
# ORCCA
# University of Colorado 

get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(EXE_NAME ${dirName} CACHE STRING "Name of executable to be created.")

if (${CMAKE_MINOR_VERSION} EQUAL 0 AND NOT APPLE) # if true, running on a CCAR computer
	set(CMAKE_C_COMPILER /usr/local/gcc/bin/gcc CACHE STRING "C Compiler" FORCE)
	set(CMAKE_CXX_COMPILER /usr/local/gcc/bin/g++ CACHE STRING "C++ Compiler" FORCE)
else()
	message("This is not a CCAR computer")
endif()

# Check what version of the compiler is installed
execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
                OUTPUT_VARIABLE GCC_VERSION)
message("GCC version is " ${GCC_VERSION})

if (GCC_VERSION VERSION_LESS 4.7) # if true, either running on a CCAR computer or a Mac (Clang has messed-up version IDs)
		if (NOT APPLE) # If NOT APPLE, then running on a CCAR computer
        	set(CMAKE_CXX_FLAGS "-std=c++0x")
        else() # Else running on a MAC. Will attempt to switch compiler to get the OMP-friendly GCC from Homebrew
        	if(EXISTS /usr/local/Cellar/gcc/7.1.0/bin/) 
	        	set(CMAKE_C_COMPILER /usr/local/Cellar/gcc/7.1.0/bin/gcc-7 CACHE STRING "C Compiler" FORCE)
				set(CMAKE_CXX_COMPILER /usr/local/Cellar/gcc/7.1.0/bin/g++-7 CACHE STRING "C++ Compiler" FORCE)
				set(CMAKE_CXX_FLAGS "-std=c++17")
			endif()
        endif()
    else() # Running on a proper Linux machine
    	set(CMAKE_CXX_FLAGS "-std=c++11")
endif()


project(${EXE_NAME})

# Specify the version used
if (${CMAKE_MAJOR_VERSION} LESS 3)
	message(FATAL_ERROR " You are running an outdated version of CMake")
endif()

cmake_minimum_required(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.0)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/source/cmake)

# Compiler flags
add_definitions(-Wall -O2)

# Include directories
include_directories(include)
include_directories(external/Gnuplot/include)
# include_directories(external/Quickhull/include)
# include_directories(external/Quickhull/Tests)
include_directories(external/CGAL_interface/include)


# Find Boost
set(BOOST_ROOT /usr/local/Cellar/boost/1.63.0/)
find_package(Boost COMPONENTS filesystem system REQUIRED) 
include_directories(${Boost_INCLUDE_DIRS}) 


# Find Armadillo 
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Find RBK 
find_package(RigidBodyKinematics REQUIRED)
include_directories(${RBK_INCLUDE_DIRS})

# Find CGAL
find_package(CGAL REQUIRED)
include( ${CGAL_USE_FILE} )
include( CGAL_CreateSingleSourceCGALProgram )


# Find Eigen3
find_package(Eigen3 3.1.0 REQUIRED)
include( ${EIGEN3_USE_FILE} )


# Find OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# Fortran compiler, required by Armadillo on Linux/Ubuntu
if(UNIX AND NOT APPLE AND ${CMAKE_MINOR_VERSION} GREATER 0)
	find_library(GFORTRAN_LIBRARY gfortran
	    PATHS /usr/lib/gcc/x86_64-linux-gnu/5/ /usr/lib/gcc/x86_64-redhat-linux/4.4.7/32/)
	list(APPEND ARMADILLO_LIBRARIES "${GFORTRAN_LIBRARY}")
endif()

# Add source files in root directory
add_executable(${EXE_NAME}
source/main.cpp
source/ShapeModel.cpp
source/ShapeModelImporter.cpp
source/Facet.cpp
source/Edge.cpp
source/Vertex.cpp
source/Lidar.cpp
source/Ray.cpp
source/RefFrame.cpp
source/FrameGraph.cpp
source/Filter.cpp
source/RK.cpp
source/DynamicAnalyses.cpp
source/Wrappers.cpp
source/Interpolator.cpp
source/BBox.cpp
source/KDTree_Shape.cpp
source/ICP.cpp
source/PC.cpp
source/PointNormal.cpp
source/KDTree_pc.cpp
include/ShapeModel.hpp
include/ShapeModelImporter.hpp
include/Facet.hpp
include/Edge.hpp
include/Vertex.hpp
include/Lidar.hpp
include/Ray.hpp
include/RefFrame.hpp
include/Adjacency_List.hpp
include/FrameGraph.hpp
include/Filter.hpp
include/RK.hpp
include/DynamicAnalyses.hpp
include/Wrappers.hpp
include/Args.hpp
include/Interpolator.hpp
include/BBox.hpp
include/KDTree_Shape.hpp
include/ICP.hpp
include/PC.hpp
include/FilterArguments.hpp
include/PointNormal.hpp
include/Constants.hpp
include/ICPException.hpp
external/Gnuplot/source/GNUPlot.cpp
external/Gnuplot/include/GNUPlot.h
external/CGAL_interface/include/CGAL_interface.hpp
external/CGAL_interface/source/CGAL_interface.cpp
)

# Linking
set(library_dependencies
${ARMADILLO_LIBRARIES}
${Boost_LIBRARIES}
${RBK_LIBRARY}
${CGAL_LIBRARIES} 
${CGAL_3RD_PARTY_LIBRARIES}
)


target_link_libraries(${EXE_NAME} ${library_dependencies})

