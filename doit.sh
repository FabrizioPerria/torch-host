#!/bin/zsh

cmake -S. -Bbuild
cmake --build build -j8 --clean-first
