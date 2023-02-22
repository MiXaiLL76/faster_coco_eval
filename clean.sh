#!/bin/bash

_dirs=$(ls)

# Удалил временные файлы python
for _dir in ${_dirs[@]}; do
  echo ${_dir}
  sudo rm -rf $(find ${_dir} -name "*__pycache__")
  sudo rm -rf $(find ${_dir} -name "*ipynb_checkpoints")
  sudo rm -rf $(find ${_dir} -name "*.DS_Store")
  
  python_files=$(find ${_dir} -name "*.py")
  for _file in ${python_files[@]}; do
    autopep8 --in-place ${_file}
  done
done

# Удалить все докера
docker images | grep faster_coco_eval | awk '{system("docker rmi " "'"faster_coco_eval:"'" $2)}'