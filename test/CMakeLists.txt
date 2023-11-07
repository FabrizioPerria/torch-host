cmake_minimum_required(VERSION 3.22)

macro(ADD_PREFIX_TO_LIST input_list prefix output_list)
    set(${output_list})
    foreach(item ${${input_list}})
        list(APPEND ${output_list} "${prefix}/${item}")
    endforeach()
endmacro()

project(YourPluginNameTest VERSION 0.1.0) #TODO: Change this to your test project name

set(LIBS_TO_TEST "YourPluginName") #TODO: Change this to your plugin name(must be same as the actuaal plugin's name)

enable_testing()
add_executable(${PROJECT_NAME})

get_target_property(JUCE_HEADER YourPluginName JUCE_LIBRARY_CODE) #TODO: Change this to your plugin name(must be same as the actuaal plugin's name)
get_target_property(JUCE_BINARY_DATA_FOLDER YourPluginName JUCE_BINARY_DATA_FOLDER) #TODO: Change this to your plugin name(must be same as the actuaal plugin's name)
target_sources(${PROJECT_NAME}
    PRIVATE
    source/AudioProcessorTest.cpp
)
ADD_PREFIX_TO_LIST(LIBS_TO_TEST "${CMAKE_CURRENT_SOURCE_DIR}/include" INCLUDE_LIB_DIRS)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${INCLUDE_LIB_DIRS}
        ${JUCE_SOURCE_DIR}/modules
        ${JUCE_HEADER}
        ${JUCE_BINARY_DATA_FOLDER}
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        ${LIBS_TO_TEST}
        GTest::gtest_main
        gmock
)

target_compile_definitions(${PROJECT_NAME}
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_SILENCE_XCODE_15_LINKER_WARNING=1
)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
