#!/bin/bash
PATH=/home/kodamak8/miniconda3/bin:/home/kodamak8/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

source /home/kodamak8/miniconda3/bin/activate && conda activate /home/kodamak8/miniconda3/envs/grid_temp

python -W ignore /home/hawaii_climate_products_container/preliminary/air_temp/code/county_map_wrapper.py

python -W ignore /home/hawaii_climate_products_container/preliminary/air_temp/code/meta_data_wrapper.py

python -W ignore /home/hawaii_climate_products_container/preliminary/air_temp/code/state_wrapper.py
