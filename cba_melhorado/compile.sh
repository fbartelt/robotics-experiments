# c++ -O3 -Wall -shared -fPIC $(python3 -m pybind11 --includes) -I/usr/include/eigen3 bind.cpp -o adaptive_cpp$(python3 -m pybind11 --extension-suffix)
# Check if build directory exists, if not create it. Else remove all files in it.
if [ ! -d "build" ]; then
    mkdir build
else
    rm -rf build/*
fi

cd build
cmake ..
make -j4
