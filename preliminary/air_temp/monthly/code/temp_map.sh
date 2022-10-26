#!/bin/bash
PATH=/home/kodamak8/miniconda3/bin:/home/kodamak8/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
export PATH
source /home/kodamak8/miniconda3/bin/activate && conda activate /home/kodamak8/miniconda3/envs/grid_temp
python -W ignore /home/hawaii_climate_products_container/preliminary/air_temp/monthly/code/monthly_map_wget.py

python -W ignore /home/hawaii_climate_products_container/preliminary/air_temp/monthly/code/update_monthly_predictor.py

python -W ignore /home/hawaii_climate_products_container/preliminary/air_temp/monthly/code/monthly_map_wrapper.py

python -W ignore /home/hawaii_climate_products_container/preliminary/air_temp/monthly/code/monthly_meta_wrapper.py

python -W ignore /home/hawaii_climate_products_container/preliminary/air_temp/monthly/code/monthly_state_wrapper.py

python -W ignore /home/hawaii_climate_products_container/preliminary/air_temp/monthly/code/monthly_stn_data.py
