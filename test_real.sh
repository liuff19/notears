#!/bin/bash


# mode=1
# batch=853
# python nonlinear_w.py --run_mode ${mode} --batch_size ${batch}

mode=1
batch=853
iter_mod=1
python nonlinear_w.py --run_mode ${mode} --batch_size ${batch} --iter_mod ${iter_mod}
