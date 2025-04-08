#!/bin/env bash

pip install matplotlib numpy pandas tensorflow kaggle
mkdir -p ~/.kaggle
echo '{"username":"tvuser","key":"d53f35c01dcc2be525eba8c0c36b9378"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c histopathologic-cancer-detection
unzip histopathologic-cancer-detection.zip
rm histopathologic-cancer-detection.zip
