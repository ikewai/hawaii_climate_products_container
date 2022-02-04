#!/bin/bash
PATH=/home/kodamak8/miniconda3/bin:/home/kodamak8/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

source /home/kodamak8/miniconda3/bin/activate && conda activate /home/kodamak8/miniconda3/envs/tempy 

python -W ignore /home/hawaii_climate_products_container/preliminary/data_aqs/code/madis/mesonet_24hr_fetch_dev.py

python -W ignore /home/hawaii_climate_products_container/preliminary/data_aqs/code/madis/hfmetar_24hr_fetch_dev.py
