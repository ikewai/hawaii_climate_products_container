import rasterio
import pytz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

MASTER_DIR = r'/home/hawaii_climate_products_container/preliminary/'
DEP_MASTER_DIR = MASTER_DIR + r'air_temp/daily/dependencies/'
PRED_DIR = DEP_MASTER_DIR + r'predictors/'
DEM_DIR = DEP_MASTER_DIR + r'geoTiffs_250m/dem/'
RUN_MASTER_DIR = MASTER_DIR + r'air_temp/data_outputs/tables/station_data/daily/raw/statewide/'
MASTER_LINK = r'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'
VAR_LIST = ['Tmax']
ISL_DICT = {'BI':['BI'],'KA':['KA'],'MN':['MA','MO','LA','KO'],'OA':['OA']}

hst = pytz.timezone('HST')
today = datetime.today().astimezone(hst)
prev_day = today - timedelta(days=1)
prev_day_monyr = prev_day.strftime('%Y_%m')

master_df = pd.read_csv(MASTER_LINK)
master_df = master_df.set_index('SKN')
meta_cols = list(master_df.columns)
for varname in VAR_LIST:
    old_pred = PRED_DIR + varname.lower() + '_predictors.csv'
    old_pred_df = pd.read_csv(old_pred)
    old_pred_df = old_pred_df.set_index('SKN')
    old_inds = old_pred_df.index.values
    pred_cols = old_pred_df.columns

    new_station_file = RUN_MASTER_DIR + '_'.join(('daily',varname,prev_day_monyr)) + '.csv'
    new_station_data = pd.read_csv(new_station_file)
    new_station_data = new_station_data.set_index('SKN')
    all_stn_inds = new_station_data.index.values

    new_stns = np.setdiff1d(all_stn_inds,old_inds)
    
    updated_inds = np.union1d(new_stns,old_inds)
    updated_preds = pd.DataFrame(index=updated_inds,columns=pred_cols)
    updated_preds.index.name = 'SKN'
    #Backfill older predictor data
    updated_preds.loc[old_inds,pred_cols] = old_pred_df

    #For each latlon in new stations list, get dem from raster based on islands
    #for each island, get skns related and get the lat lons at that skn
    #determine which new stations are on specific island
    for isl in list(ISL_DICT.keys()):
        isl_raster_file = DEM_DIR + '_'.join((isl.lower(),'dem','250m')) + '.tif'
        isl_raster = rasterio.open(isl_raster_file)
        isl_raster_dem = isl_raster.read(1)
        isl_list = ISL_DICT[isl]
        new_stns_meta = master_df.loc[new_stns]
        #Df of metadata corresponding to new stations sorted by island
        isl_new_stns = new_stns_meta[new_stns_meta['Island'].isin(isl_list)]
        isl_new_latlons = [(skn,isl_new_stns.at[skn,'LON'],isl_new_stns.at[skn,'LAT']) for skn in isl_new_stns.index.values]
        isl_raster_coords = [(skn,)+isl_raster.index(i,j) for (skn,i,j) in isl_new_latlons]
        #wait need to make sure they're still associated with skn
        #So have skn, and raster index. Need actual dem value 
        isl_new_dems = [(skn,isl_raster_dem[i,j]) for (skn,i,j) in isl_raster_coords]
        idx,values = zip(*isl_new_dems)
        updated_preds.loc[idx,'dem_250'] = values
    
    #match other metadata
    match_cols = [col for col in list(updated_preds.columns) if col in meta_cols]
    updated_preds.loc[new_stns,match_cols] = master_df.loc[new_stns,match_cols]
    #After all dems have been updated in file, rewrite the file
    updated_preds = updated_preds.reset_index()
    updated_preds.to_csv(old_pred,index=False)



