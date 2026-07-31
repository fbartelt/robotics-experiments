# !/bin/bash
set -e

SRC_PATH="$(pwd)"
INSTALL_PREFIX="${SRC_PATH}/third_party/install"

echo "Building libcdd"
cd ./third_party/libccd
if [ ! -d "build" ]; then
    mkdir build
else
    rm -rf build/*
fi
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
make -j4
make install
cd "$SRC_PATH"

echo "Building fcl"
cd ./third_party/fcl
if [ ! -d "build" ]; then
    mkdir build
else
    rm -rf build/*
fi
cd build
cmake .. -DFCL_WITH_OCTOMAP=OFF -DFCL_STATIC_LIBRARY=ON -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
make -j4
make install

cd "$SRC_PATH"
echo "All done! Libraries are in ${INSTALL_PREFIX}"
