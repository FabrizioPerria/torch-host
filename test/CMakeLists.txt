cmake_minimum_required(VERSION 3.22)

macro(ADD_PREFIX_TO_LIST input_list prefix output_list)
    set(${output_list})
    foreach(item ${${input_list}})
        list(APPEND ${output_list} "${prefix}/${item}")
    endforeach()
endmacro()

project(YourPluginNameTest VERSION 0.1.0)

set(LIBS_TO_TEST "")
list(APPEND LIBS_TO_TEST "YourPluginName")

enable_testing()

juce_add_console_app(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PRIVATE
    source/AudioProcessorTest.cpp
)
ADD_PREFIX_TO_LIST(LIBS_TO_TEST "${CMAKE_CURRENT_SOURCE_DIR}/include" INCLUDE_LIB_DIRS)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${INCLUDE_LIB_DIRS}
        ${JUCE_SOURCE_DIR}/modules
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        ${LIBS_TO_TEST}
        GTest::gtest_main
)

juce_generate_juce_header(${PROJECT_NAME})

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
