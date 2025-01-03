build_folder="build"

if [ ! -d $build_folder ]; then
    mkdir -p $build_folder
    pushd $build_folder
    cmake ..
    popd
fi

pushd build
make
popd