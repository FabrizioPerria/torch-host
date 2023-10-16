cmake_minimum_required(VERSION 3.22)

project(AudioPlugin VERSION 0.1.0)

juce_add_plugin(${PROJECT_NAME}
    COMPANY xx
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    PLUGIN_MANUFACTURER_CODE XXXX
    PLUGIN_CODE AUDI
    FORMATS VST3 AU Standalone
    PRODUCT_NAME "YourPluginName"
)

target_sources(${PROJECT_NAME}
    PRIVATE
        source/PluginEditor.cpp
        source/PluginProcessor.cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        include/YourPluginName
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        juce::juce_dsp
        juce::juce_audio_utils
        juce::juce_gui_basics
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

target_compile_definitions(${PROJECT_NAME}
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_SILENCE_XCODE_15_LINKER_WARNING=1
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_link_options(${PROJECT_NAME} PUBLIC "-Wl,-weak_reference_mismatches,weak")

juce_generate_juce_header(${PROJECT_NAME})
