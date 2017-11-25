#!/bin/bash

docker build -t image-converter:latest --cache-from alpine-python3-opencv:latest -f ./Dockerfile ../..
