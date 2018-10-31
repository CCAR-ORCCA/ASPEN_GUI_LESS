# CMakeLists.txt for ShapeUncertainty
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
	set(OC_LOC "/home/bebe0705/libs/local/lib/cmake/OrbitConversions")
	set(SBGAT_LOC "/home/bebe0705/libs/local/lib/cmake/SbgatCore")
	set(ASPEN_LOC "/home/bebe0705/libs/local/lib/cmake/ASPEN")
	set(CGAL_interface_LOC "/home/bebe0705/libs/local/lib/cmake/CGAL_interface")
	set (VTK_PATH /usr/local/VTK-8.1.0/lib/cmake/vtk-8.1)
elseif(UNIX AND NOT APPLE)
	set(IS_FORTUNA ON)
	set(RBK_LOC "/usr/local/lib/cmake/RigidBodyKinematics")
	set(SBGAT_LOC "/home/bebe0705/libs/local/lib/cmake/SbgatCore")
	set(ASPEN_LOC "/usr/local/lib/cmake/ASPEN")
	set(CGAL_interface_LOC "/usr/local/lib/cmake/CGAL_interface")
	set (VTK_PATH /home/ben/Work/VTK-no-QT/build)
endif()

cmake_minimum_required(VERSION 3.0.0)


if (${USE_GCC})
	include(cmake/FindOmpGcc.cmake)
else()
	set(CMAKE_C_COMPILER /usr/bin/gcc CACHE STRING "C Compiler" FORCE)
	set(CMAKE_CXX_COMPILER /usr/bin/g++ CACHE STRING "C++ Compiler" FORCE)
endif()


# Building procedure
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(EXE_NAME ${dirName} CACHE STRING "Name of executable to be created.")


project(${EXE_NAME})

# Specify the version used
if (${CMAKE_MAJOR_VERSION} LESS 3)
	message(FATAL_ERROR " You are running an outdated version of CMake")
endif()


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/source/cmake)

# Compiler flags
add_definitions(-Wall -O2 )



# Enable C++17 
if (EXISTS /home/bebe0705/.am_fortuna)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fext-numeric-literals")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif()

# Find ASPEN
find_package(ASPEN REQUIRED PATHS ${ASPEN_LOC}) 
include_directories(${ASPEN_INCLUDE_HEADER}) 
include_directories(${ASPEN_INCLUDE_GNUPLOT}) 

# Find Boost
find_package(Boost COMPONENTS filesystem system REQUIRED) 
include_directories(${Boost_INCLUDE_DIRS}) 


# Find Armadillo 
find_package(Armadillo REQUIRED )
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Find RBK 
find_package(RigidBodyKinematics REQUIRED PATHS ${RBK_LOC})
include_directories(${RBK_INCLUDE_DIR})


# Find RBK 
find_package(OrbitConversions REQUIRED PATHS ${OC_LOC})
include_directories(${OC_INCLUDE_DIR})


# Find VTK Package
find_package(VTK REQUIRED PATHS ${VTK_PATH})
include(${VTK_USE_FILE})

# Find CGAL
find_package(CGAL REQUIRED)
include( ${CGAL_USE_FILE} )
include( CGAL_CreateSingleSourceCGALProgram )

# Find CGAL interface
find_package(CGAL_interface REQUIRED PATHS ${CGAL_interface_LOC})
include_directories( ${CGAL_interface_INCLUDE_DIR} )

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
# if(UNIX AND NOT APPLE AND ${CMAKE_MINOR_VERSION} GREATER 0 AND NOT ${IS_FORTUNA})
# 	find_library(GFORTRAN_LIBRARY gfortran
# 	    PATHS /usr/lib/gcc/x86_64-linux-gnu/5/ /usr/lib/gcc/x86_64-redhat-linux/4.4.7/32/)
# 	list(APPEND ARMADILLO_LIBRARIES "${GFORTRAN_LIBRARY}")
# endif()

# Add source files in root directory
add_executable(${EXE_NAME}
	main.cpp)


# Linking
set(library_dependencies
	${ARMADILLO_LIBRARIES}
	${Boost_LIBRARIES}
	${RBK_LIBRARY}
	${OC_LIBRARY}
	${CGAL_LIBRARIES} 
	${CGAL_3RD_PARTY_LIBRARIES}
	${VTK_LIBRARIES}
	${SBGATCORE_LIBRARY}
	${CGAL_interface_LIBRARY}
	${ASPEN_LIBRARY}
	)

if (UNIX AND NOT APPLE)

	target_link_libraries(${EXE_NAME} ${library_dependencies})
elseif (OPENMP_FOUND)
	target_link_libraries(${EXE_NAME} ${library_dependencies} OpenMP::OpenMP_CXX)

else()
	target_link_libraries(${EXE_NAME} ${library_dependencies})

endif()

