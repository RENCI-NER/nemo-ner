docker run --gpus all -it --rm -v $PWD:/src  -v $PWD/../NeMo/:/NeMo --shm-size=8g \
-p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
stack=67108864 --device=/dev/snd nvcr.io/nvidia/nemo:1.4.0