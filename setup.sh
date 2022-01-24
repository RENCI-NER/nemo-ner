#!/usr/bin/env bash
# create conda env


set -e

conda init bash

conda create --name nemo-1.4.0 python=3.8.10

conda activate nemo-1.4.0

pip install -r ./pre-requirements.txt

pip install -r requirements.txt