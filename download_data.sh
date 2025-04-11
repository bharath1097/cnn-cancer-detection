#!/bin/bash

# TODO Provide alternate path for downloading data
#      Hide the kaggle credentials in a config file (with dummy sample)
mkdir -p ~/.kaggle
echo '{"username":"tvuser","key":"d53f35c01dcc2be525eba8c0c36b9378"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c histopathologic-cancer-detection
unzip histopathologic-cancer-detection.zip
rm histopathologic-cancer-detection.zip
