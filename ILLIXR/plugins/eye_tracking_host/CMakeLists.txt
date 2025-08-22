set(PLUGIN_NAME plugin.eye_tracking_host${ILLIXR_BUILD_SUFFIX})

# link_directories(/usr/local/lib)


find_package(onnxruntime REQUIRED)
find_package(CUDNN REQUIRED)

# source files, listed individually so that any changes will trigger a rebuild
add_library(${PLUGIN_NAME} SHARED plugin.cpp
        ../../include/illixr/plugin.hpp
        ../../include/illixr/opencv_data_types.hpp
        ../../include/illixr/data_format.hpp
        ../../include/illixr/phonebook.hpp
        ../../include/illixr/eye_tracking_host.hpp
        )

if(BUILD_OPENCV)
        add_dependencies(${PLUGIN_NAME} OpenCV_Viz)
endif()

target_include_directories(${PLUGIN_NAME} PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR} 
        ${OpenCV_INCLUDE_DIRS} 
        ${Protobuf_INCLUDE_DIRS} 
        ${CUDNN_INCLUDE_DIR}
        ${onnxruntime_INCLUDE_DIRS}
        ${ILLIXR_SOURCE_DIR}/include)

target_include_directories(${PLUGIN_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} ${ILLIXR_SOURCE_DIR}/include ${Eigen3_INCLUDE_DIRS})
target_link_libraries(${PLUGIN_NAME} 
        ${CUDNN_LIBRARY}
        ${OpenCV_LIBRARIES}
        ${Eigen3_LIBRARIES}
        ${onnxruntime_LIBRARIES}
        spdlog::spdlog)

target_compile_features(${PLUGIN_NAME} PRIVATE cxx_std_17)

install(TARGETS ${PLUGIN_NAME} DESTINATION lib)