build_folder="build"

if [ ! -d $build_folder ]; then
    mkdir -p $build_folder
fi

pushd $build_folder
    cmake -DOpenCV_DIR=/c/lib/install/opencv/ ..
    cmake --build .
popd