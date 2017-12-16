# @file   CMakeLists.txt
# @Author Benjamin Bercovici (bebe0705@colorado.edu)
# @date   July, 2017
# @brief  CMake listing enabling compilation and installation of the RBK library


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


# Building procedure
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(LIB_NAME ASPEN CACHE STRING "Name of library to be created.")

if (${CMAKE_MINOR_VERSION} EQUAL 0 AND NOT APPLE) # if true, running on a CCAR computer (old CMAKE and RedHat)
	set(CMAKE_C_COMPILER /usr/local/gcc/bin/gcc CACHE STRING "C Compiler" FORCE)
	set(CMAKE_CXX_COMPILER /usr/local/gcc/bin/g++ CACHE STRING "C++ Compiler" FORCE)
elseif(NOT APPLE)
	# Not sure about this. Should work on a modern Ubuntu system
	set(CMAKE_C_COMPILER /usr/bin/gcc CACHE STRING "C Compiler" FORCE)
	set(CMAKE_CXX_COMPILER /usr/bin/g++ CACHE STRING "C++ Compiler" FORCE)
    set (CMAKE_CXX_STANDARD 17)
	message("This is a Linux machine. CMAKE_CXX_STANDARD 17 used")


endif()

# Check what version of the compiler is installed
execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
                OUTPUT_VARIABLE GCC_VERSION)
message("Default compiler is " ${CMAKE_C_COMPILER})
message("Default ${CMAKE_C_COMPILER} version is " ${GCC_VERSION})

# Set up the C++11 and OMP compile flags 
if (GCC_VERSION VERSION_LESS 4.7 OR APPLE) # if true, either running on a CCAR computer or a Mac (Clang has messed-up version IDs)
		if (NOT APPLE) # If NOT APPLE, then running on a CCAR computer. The proper compiler should have already been set
        	set(CMAKE_CXX_FLAGS "-std=c++0x")
        elseif(EXISTS ${OMP_FRIENDLY_GCC_PATH}) # Else running on a MAC. Will attempt to switch compiler to get the OMP-friendly GCC 6.3.0_1 from Homebrew
        		message("Switching to OMP-friendly GCC ")
	        	set(CMAKE_C_COMPILER ${OMP_FRIENDLY_GCC_PATH}gcc-${OMP_FRIENDLY_GCC_MAJOR_VERSION} CACHE STRING "C Compiler" FORCE)
				set(CMAKE_CXX_COMPILER ${OMP_FRIENDLY_GCC_PATH}g++-${OMP_FRIENDLY_GCC_MAJOR_VERSION} CACHE STRING "C++ Compiler" FORCE)
				set(CMAKE_CXX_FLAGS "-std=c++17")
			
			else()
				message(FATAL_ERROR "No OMP-compliant compiler was found on this mac. ")
		endif()
endif()


project(${LIB_NAME})



# Specify the version used
if (${CMAKE_MAJOR_VERSION} LESS 3)
	message(FATAL_ERROR " You are running an outdated version of CMake")
endif()


cmake_minimum_required(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.0)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/source/cmake)

# Compiler flags
add_definitions(-Wall -O2)

# Set path to include and source folders
set(INCLUDES_PATH /Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/lib/include)
set(EXTERNAL_PATH /Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/lib/external)
set(SOURCE_PATH /Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/lib/source)

# Include directories
include_directories(${INCLUDES_PATH})
include_directories(${EXTERNAL_PATH}/Gnuplot/include)
include_directories(${EXTERNAL_PATH}/CGAL_interface/include)

# Find Boost
find_package(Boost COMPONENTS filesystem system REQUIRED) 
include_directories(${Boost_INCLUDE_DIRS}) 


# Find Armadillo 
find_package(Armadillo REQUIRED )
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Find RBK 
find_package(RigidBodyKinematics REQUIRED)
include_directories(${RBK_INCLUDE_DIRS})

# Find CGAL
find_package(CGAL REQUIRED)
include( ${CGAL_USE_FILE} )
include( CGAL_CreateSingleSourceCGALProgram )


# Find SBGAT 
find_package(SbgatCore REQUIRED)
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
if(UNIX AND NOT APPLE AND ${CMAKE_MINOR_VERSION} GREATER 0)
	find_library(GFORTRAN_LIBRARY gfortran
	    PATHS /usr/lib/gcc/x86_64-linux-gnu/5/ /usr/lib/gcc/x86_64-redhat-linux/4.4.7/32/)
	list(APPEND ARMADILLO_LIBRARIES "${GFORTRAN_LIBRARY}")
endif()

# Add source files in root directory
add_library(${LIB_NAME} SHARED
${SOURCE_PATH}/BatchFilter.cpp
${SOURCE_PATH}/BBox.cpp
${SOURCE_PATH}/Bezier.cpp
${SOURCE_PATH}/ControlPoint.cpp
${SOURCE_PATH}/DynamicAnalyses.cpp
${SOURCE_PATH}/Element.cpp
${SOURCE_PATH}/Facet.cpp
${SOURCE_PATH}/Filter.cpp
${SOURCE_PATH}/FrameGraph.cpp
${SOURCE_PATH}/ICP.cpp
${SOURCE_PATH}/Interpolator.cpp
${SOURCE_PATH}/KDTree_pc.cpp
${SOURCE_PATH}/KDTree_shape.cpp
${SOURCE_PATH}/KDTree_control_points.cpp
${SOURCE_PATH}/Lidar.cpp
${SOURCE_PATH}/PC.cpp
${SOURCE_PATH}/PointNormal.cpp
${SOURCE_PATH}/Ray.cpp
${SOURCE_PATH}/RefFrame.cpp
${SOURCE_PATH}/ShapeBuilder.cpp
${SOURCE_PATH}/ShapeFitter.cpp
${SOURCE_PATH}/ShapeFitterBezier.cpp
${SOURCE_PATH}/ShapeFitterTri.cpp
${SOURCE_PATH}/ShapeModel.cpp
${SOURCE_PATH}/ShapeModelBezier.cpp
${SOURCE_PATH}/ShapeModelImporter.cpp
${SOURCE_PATH}/ShapeModelTri.cpp
${SOURCE_PATH}/RK.cpp
${SOURCE_PATH}/Wrapper.cpp
${EXTERNAL_PATH}/Gnuplot/source/GNUPlot.cpp
${EXTERNAL_PATH}/CGAL_interface/source/CGAL_interface.cpp
)

# Linking
set(library_dependencies
${ARMADILLO_LIBRARIES}
${Boost_LIBRARIES}
${RBK_LIBRARY}
${CGAL_LIBRARIES} 
${CGAL_3RD_PARTY_LIBRARIES}
${SBGATCORE_LIBRARY}
)


target_link_libraries(${LIB_NAME} ${library_dependencies})

file(GLOB includes "${INCLUDES_PATH}/*.hpp")
file(GLOB includes_gnuplot "${EXTERNAL_PATH}/Gnuplot/include/*.hpp")
file(GLOB includes_cgal "${EXTERNAL_PATH}/CGAL_interface/include/*.hpp")

install (TARGETS ${LIB_NAME} DESTINATION /usr/local/lib)
install(FILES ${includes} DESTINATION /usr/local/include/ASPEN/)
install(FILES ${includes_gnuplot} DESTINATION /usr/local/include/ASPEN/external/Gnuplot/)
install(FILES ${includes_cgal} DESTINATION /usr/local/include/ASPEN/external/CGAL_interface/)

install (FILES ${PROJECT_SOURCE_DIR}/ASPENConfig.cmake DESTINATION /usr/local/lib/cmake/ASPEN/)


