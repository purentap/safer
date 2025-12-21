#!/bin/bash

SEEDS="0,1,2"

python train.py -m \
    seed=${SEEDS}