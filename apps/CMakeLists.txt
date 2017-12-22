include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../kfusion/include/)

##########################################################
######################## BOOST ###########################
##########################################################

find_package(Boost COMPONENTS system filesystem REQUIRED)
if(Boost_FOUND)
    message("-- Boost ${Boost_VERSION} was found")
    include_directories(${Boost_INCLUDE_DIRS})
else(BOOST_FOUND)
    message(FATAL_ERROR "Boost was not found")
endif(Boost_FOUND)

##########################################################
######################## C++11 ###########################
##########################################################

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    message(STATUS "-std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    message(STATUS "-std=c++0x")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

##########################################################
######################## BUILD ###########################
##########################################################

# APP 01: OpenNI capture
if (OPENNI_FOUND)
    add_executable(demo demo.cpp)
    target_link_libraries(demo ${OpenCV_LIBS} ${Boost_LIBRARIES} kfusion)

    set_target_properties(demo PROPERTIES
        DEBUG_POSTFIX "d"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    install(TARGETS demo RUNTIME DESTINATION bin COMPONENT main)
    install(FILES demo.cpp DESTINATION app COMPONENT main)
endif(OPENNI_FOUND)

# APP 02: BinGrabber capture
add_executable(demo_bin demo_bin.cpp)
target_link_libraries(demo_bin ${OpenCV_LIBS} ${PCL_LIBRARIES} kfusion)

set_target_properties(demo_bin PROPERTIES
        DEBUG_POSTFIX "d"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

##########################################################
##################### INSTALLATION #######################
##########################################################

install(TARGETS demo_bin RUNTIME DESTINATION bin COMPONENT main)
install(FILES demo_bin.cpp DESTINATION app COMPONENT main)
