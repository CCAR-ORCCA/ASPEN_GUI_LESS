# CMakeLists.txt for ShapeModelBezier
# Benjamin Bercovici, 11/10/2017
# ORCCA
# University of Colorado 



################################################################################
#
# 								User-defined paths
#						Should be checked for consistency
#						Before running 'cmake ..' in build dir
#
################################################################################

# OMP-friendly GCC Major version (will only be used on Mac OS systems)
set (OMP_FRIENDLY_GCC_MAJOR_VERSION 7)

# Path to OMP-Friendly GCC Compiler (will only be used on Mac OS systems)
set (OMP_FRIENDLY_GCC_PATH /usr/local/Cellar/gcc/7.2.0/bin/)

################################################################################
#
#
# 		The following should normally not require any modification
# 				Unless new files are added to the build tree
#
#
################################################################################

if (EXISTS /home/bebe0705/.am_fortuna)
	set(IS_FORTUNA ON)
	set(RBK_LOC "/home/bebe0705/libs/local/lib/cmake/RigidBodyKinematics")
	set(SBGAT_LOC "/home/bebe0705/libs/local/lib/cmake/SbgatCore")
	set(ASPEN_LOC "/home/bebe0705/libs/local/lib/cmake/ASPEN")

else()
	set(IS_FORTUNA OFF)
	message("-- This is not Fortuna")

endif()


# Building procedure
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(EXE_NAME ${dirName} CACHE STRING "Name of executable to be created.")

if (${IS_FORTUNA})
	set(CMAKE_C_COMPILER "/usr/local/bin/gcc" CACHE STRING "C Compiler" FORCE)
	set(CMAKE_CXX_COMPILER "/usr/local/bin/g++" CACHE STRING "C++ Compiler" FORCE)
else()
	if(APPLE)
		if(EXISTS ${OMP_FRIENDLY_GCC_PATH}) # Else running on a MAC. Will attempt to switch compiler to get the OMP-friendly GCC 6.3.0_1 from Homebrew
			message("Switching to OMP-friendly GCC ")
	    	set(CMAKE_C_COMPILER ${OMP_FRIENDLY_GCC_PATH}gcc-${OMP_FRIENDLY_GCC_MAJOR_VERSION} CACHE STRING "C Compiler" FORCE)
			set(CMAKE_CXX_COMPILER ${OMP_FRIENDLY_GCC_PATH}g++-${OMP_FRIENDLY_GCC_MAJOR_VERSION} CACHE STRING "C++ Compiler" FORCE)

		else()# no OMP-compliant compiler was found on this mac. 
			message(FATAL_ERROR "No OMP-compliant compiler was found on this Mac.")
		endif()
	else() # Running on Linux. Will switch back to compiler in /usr/local/bin
				message("Switching to /usr/local/gcc ")
		    	set(CMAKE_C_COMPILER "/usr/local/bin/gcc" CACHE STRING "C Compiler" FORCE)
				set(CMAKE_CXX_COMPILER "/usr/local/bin/g++" CACHE STRING "C++ Compiler" FORCE)
	endif()
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")


# Find ASPEN
find_package(ASPEN REQUIRED PATHS ${ASPEN_LOC}) 
include_directories(${ASPEN_INCLUDE_HEADER}) 
include_directories(${ASPEN_INCLUDE_GNUPLOT}) 
include_directories(${ASPEN_INCLUDE_CGAL}) 

# Find Boost
find_package(Boost COMPONENTS filesystem system REQUIRED) 
include_directories(${Boost_INCLUDE_DIRS}) 


# Find Armadillo 
find_package(Armadillo REQUIRED )
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Find RBK 
find_package(RigidBodyKinematics REQUIRED PATHS ${RBK_LOC})
include_directories(${RBK_INCLUDE_DIRS})

# Find CGAL
find_package(CGAL REQUIRED)
include( ${CGAL_USE_FILE} )
include( CGAL_CreateSingleSourceCGALProgram )


# Find SBGAT 
find_package(SbgatCore REQUIRED PATHS ${SBGAT_LOC})
include_directories(${SBGATCORE_INCLUDE_HEADER})


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
if(UNIX AND NOT APPLE AND ${CMAKE_MINOR_VERSION} GREATER 0 AND NOT ${IS_FORTUNA})
	find_library(GFORTRAN_LIBRARY gfortran
	    PATHS /usr/lib/gcc/x86_64-linux-gnu/5/ /usr/lib/gcc/x86_64-redhat-linux/4.4.7/32/)
	list(APPEND ARMADILLO_LIBRARIES "${GFORTRAN_LIBRARY}")
endif()

# Add source files in root directory
add_executable(${EXE_NAME}
main.cpp)

# Linking
set(library_dependencies
${ARMADILLO_LIBRARIES}
${Boost_LIBRARIES}
${RBK_LIBRARY}
${CGAL_LIBRARIES} 
${CGAL_3RD_PARTY_LIBRARIES}
${SBGATCORE_LIBRARY}
${ASPEN_LIBRARY}
)


target_link_libraries(${EXE_NAME} ${library_dependencies})

