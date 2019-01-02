project(GeoReference)
cmake_minimum_required(VERSION 2.8)

set(FOLDER_NAME "tools")
set(PCL_DIR "PCL_DIR-NOTFOUND" CACHE "PCL_DIR" "Path to the pcl installation directory")
set(PROJ4_INCLUDE_DIR "/usr/include/" CACHE "PROJ4_INCLUDE_DIR" "Path to the proj4 inlcude directory")
find_package(PROJ4)

FIND_PACKAGE(Boost REQUIRED thread filesystem)
FIND_PACKAGE(PCL 1.8 HINTS "${PCL_DIR}/share/pcl-1.8")
find_library(PROJ4_LIBRARY "libproj.so" PATHS "/usr/lib" "/usr/lib/x86_64-linux-gnu")

include_directories(${PCL_ROOT}/include/pcl-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR})
include_directories(${PROJ4_INCLUDE_DIRS})

COLMAP_ADD_EXECUTABLE(GeoReference GeoReference.cc)
TARGET_LINK_LIBRARIES(GeoReference ${Boost_LIBRARIES})
target_link_libraries(GeoReference ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_SURFACE_LIBRARIES} ${PCL_FEATURES_LIBRARIES})
target_link_libraries(GeoReference ${PROJ4_LIBRARIES})
target_link_libraries(GeoReference ${PROJ4_LIBRARY})