#! /bin/bash

current_folder=/home/xinyuan/workspace/git/real_time_face_recognition

docker run --runtime=nvidia -it --rm \
    -e LC_ALL='C.UTF-8' \
    -e LANG='C.UTF-8' \
    -v ${current_folder}:/real_time_face_recognition \
    face_recognition /bin/bash