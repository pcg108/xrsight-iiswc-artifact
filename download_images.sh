#~/bin/bash

if [ ! -f qemu_images.tar.gz ] && [ ! -d temp ]; then
    wget https://xrsight-iiswc-artifact-evaluation.s3.us-east-1.amazonaws.com/qemu_images.tar.gz
    tar -xzf qemu_images.tar.gz
fi

if [ ! -f qemu-vio.tar.gz ]; then
    wget https://xrsight-iiswc-artifact-evaluation.s3.us-east-1.amazonaws.com/qemu-vio.tar.gz
    tar -xzf qemu-vio.tar.gz
    mv ubuntu-base-vio.img temp/
fi