#wget files before daily rf run TESTING 

#gap fill files
cd /home/hawaii_climate_products_container/preliminary/rainfall/dependencies/daily/gapFilling
wget https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/rainfall/HCDP_dependicies/daily/gapFilling/SKN_1006_Input.csv 
wget -A csv -m -p -E -k -K -np https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/rainfall/HCDP_dependicies/daily/gapFilling/
#wget --accept pdf,jpg --mirror --page-requisites --adjust-extension --convert-links --backup-converted --no-parent http://site/path/
wget -r --no-parent -A 'SKN_*.Input.csv' https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/rainfall/HCDP_dependicies/daily/gapFilling/

#prob rasters
cd /home/hawaii_climate_products_container/preliminary/rainfall/dependencies/daily/probRasters/6mo
wget https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/rainfall/HCDP_dependicies/daily/probRasters/6mo/hi_statewide_meanlog_6mo_mjjaso.tif 
wget -A tif -m -p -E -k -K -np https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/rainfall/HCDP_dependicies/daily/probRasters/6mo/

cd /home/hawaii_climate_products_container/preliminary/rainfall/dependencies/daily/probRasters/ann
wget https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/rainfall/HCDP_dependicies/daily/probRasters/ann/hi_statewide_sdlog_annual_all.tif 
wget -A tif -m -p -E -k -K -np https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/rainfall/HCDP_dependicies/daily/probRasters/ann/

#qaqc models
cd /home/hawaii_climate_products_container/preliminary/rainfall/dependencies/daily/models
wget https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/rainfall/HCDP_dependicies/daily/models/BI_no_human_rf_nonzero_randfor.rds
wget -A rds -m -p -E -k -K -np https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/rainfall/HCDP_dependicies/daily/models/



