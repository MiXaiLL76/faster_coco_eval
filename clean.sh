#!/bin/bash

_dirs=$(ls | grep -wv storage)

# Удалил временные файлы python
for _dir in "${_dirs[@]}"; do
  sudo rm -rf $(find ${_dir} -name "*__pycache__")
  sudo rm -rf $(find ${_dir} -name "*ipynb_checkpoints")
  sudo rm -rf $(find ${_dir} -name "*.DS_Store")
  autopep8 --in-place $(find ${_dir} -name "*.py")
done