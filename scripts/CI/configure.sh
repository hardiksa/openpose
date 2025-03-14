#!/bin/bash

# Configure the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

echo "WITH_CMAKE = ${WITH_CMAKE}."
if [[ $WITH_CMAKE == true ]] ; then
  echo "Running CMake configuration..."
  source $BASEDIR/configure_cmake.sh
else
  echo "Running Makefile configuration..."
  source $BASEDIR/configure_make.sh
fi

# Compile and run the code to see if the ONNX model generates
echo "Compiling and running the code to generate ONNX model..."
if [[ $WITH_CMAKE == true ]] ; then
  cmake --build build --target all
else
  make all
fi

# Run the example to generate the ONNX model
echo "Running the example to generate ONNX model..."
if [[ $WITH_CMAKE == true ]] ; then
  ./build/examples/tutorial_api_cpp/06_face_from_image.bin --face_net_resolution 32x32 --write_json output/ --write_images output/ --no_display
else
  ./examples/tutorial_api_cpp/06_face_from_image.bin --face_net_resolution 32x32 --write_json output/ --write_images output/ --no_display
fi
