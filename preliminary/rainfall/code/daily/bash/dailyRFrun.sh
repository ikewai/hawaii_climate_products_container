#run all daily rf r code in sequence
Rscript /home/hawaii_climate_products_container/preliminary/rainfall/code/rcode/hads_daily_rf_FINAL.R
Rscript /home/hawaii_climate_products_container/preliminary/rainfall/code/rcode/nws_rr5_daily_rf_FINAL.R
#need MADIS rf daily agg here
Rscript /home/hawaii_climate_products_container/preliminary/rainfall/code/rcode/all_data_daily_merge_table_FINAL.R
Rscript /home/hawaii_climate_products_container/preliminary/rainfall/code/rcode/qaqc_randfor_bad_data_flag_remove_FINAL.R