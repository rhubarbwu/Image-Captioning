#!/bin/sh

cd data/images

case $1 in
test2014)
    wget http://images.cocodataset.org/zips/test2014.zip
    unzip test2014.zip
    ;;
train2014)
    wget http://images.cocodataset.org/zips/train2014.zip
    unzip train2014.zip
    ;;
val2014)
    wget http://images.cocodataset.org/zips/val2014.zip
    unzip val2014.zip
    ;;
esac
