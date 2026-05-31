cmake_minimum_required(VERSION 3.22.1)

project("mediapipefacelandmark")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 获取当前 CMakeLists.txt 所在目录（支持作为子模块使用）
set(MEDIAPIPE_FACE_DIR ${CMAKE_CURRENT_LIST_DIR})

# New modular source files
set(MEDIAPIPE_FACE_SOURCES
    ${MEDIAPIPE_FACE_DIR}/src/core/face_detector.cpp
    ${MEDIAPIPE_FACE_DIR}/src/core/face_landmarker.cpp
    ${MEDIAPIPE_FACE_DIR}/src/core/blendshape_predictor.cpp
    ${MEDIAPIPE_FACE_DIR}/yuv_converter.cpp
    ${MEDIAPIPE_FACE_DIR}/mnn_yuv_converter.cpp
)

# 头文件目录（导出给父项目使用）
set(MEDIAPIPE_FACE_INCLUDE_DIR ${MEDIAPIPE_FACE_DIR}/include CACHE PATH "MediaPipe Face include directory")

# MNN 头文件目录 - 使用 ThirdParty 中的原始 MNN 仓库
# 路径: MediapipeFaceLandmark/src/main/cpp -> 工程根目录 -> ThirdParty/MNN/include
get_filename_component(PROJECT_ROOT_DIR "${MEDIAPIPE_FACE_DIR}/../../../.." ABSOLUTE)
set(MNN_INCLUDE_DIR "${PROJECT_ROOT_DIR}/ThirdParty/MNN/include" CACHE PATH "MNN include directory")

if(NOT EXISTS "${MNN_INCLUDE_DIR}/MNN")
    message(FATAL_ERROR "MNN include directory not found: ${MNN_INCLUDE_DIR}/MNN")
endif()

set(MNN_USE_LOGCAT ON)
option(MNN_USE_LOGCAT "Use Logcat instead of print for info" ON)

# 如果作为独立项目构建（非子模块）
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # 独立构建模式 - 包含 JNI 并构建为完整的 shared library
    add_library(${CMAKE_PROJECT_NAME} SHARED
            ${MEDIAPIPE_FACE_DIR}/MediaPipeFaceLandmarkJNI.cpp
            ${MEDIAPIPE_FACE_SOURCES}
    )

    # 优先使用 ThirdParty 中的原始 MNN 头文件
    include_directories(${MNN_INCLUDE_DIR} ${MEDIAPIPE_FACE_INCLUDE_DIR})

    # Add library paths (独立构建时使用本地 jniLibs)
    set(LIB_PATH "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}")

    add_library(MNN SHARED IMPORTED)
    set_target_properties(MNN PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNN.so")

    add_library(mnn_Express SHARED IMPORTED)
    set_target_properties(mnn_Express PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNN_Express.so")

    target_link_libraries(${CMAKE_PROJECT_NAME}
            android
            log
            MNN
            mnn_Express
    )
else()
    # 作为子模块被其他项目引用 - 创建 STATIC 库供父项目链接
    add_library(mediapipe_face_core STATIC
            ${MEDIAPIPE_FACE_SOURCES}
    )

    target_include_directories(mediapipe_face_core PUBLIC ${MNN_INCLUDE_DIR} ${MEDIAPIPE_FACE_INCLUDE_DIR})

    # 导出源文件列表和 JNI 文件，供父项目选择性使用
    set(MEDIAPIPE_FACE_SOURCES ${MEDIAPIPE_FACE_SOURCES} PARENT_SCOPE)
    set(MEDIAPIPE_FACE_JNI_SOURCE ${MEDIAPIPE_FACE_DIR}/MediaPipeFaceLandmarkJNI.cpp PARENT_SCOPE)
    set(MEDIAPIPE_FACE_INCLUDE_DIR ${MEDIAPIPE_FACE_INCLUDE_DIR} PARENT_SCOPE)
    set(MNN_INCLUDE_DIR ${MNN_INCLUDE_DIR} PARENT_SCOPE)
endif()