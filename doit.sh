#!/bin/zsh

cmake -S. -Bbuild
cmake --build build -j8
