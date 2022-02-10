"""
This builds a script to convert raw madis sourced csv into min and max temperature.

Version 1.0
"""

import sys
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from os.path import exists

#DEFINE CONSTANTS--------------------------------------------------------------
SOURCE = 'madis'
SRC_VARNAME = 'temperature'
SRC_KEY = 'stationId'
SRC_TIME = 'time'
SRC_VARKEY = 'varname'
MASTER_KEY = 'NWS.id'
INT_EXCEPT = {'E3941':144.,'F4600':96.}
MASTER_LINK = r'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'
MASTER_DIR = r'/home/hawaii_climate_products_container/preliminary/'
SOURCE_DIR = MASTER_DIR + r'data_aqs/data_outputs/madis/parse/'
PROC_OUTPUT_DIR = MASTER_DIR + r'air_temp/working_data/processed_data/' + SOURCE + r'/'
TRACK_DIR = MASTER_DIR + r'air_temp/data_outputs/tables/air_temp_station_tracking/'
#END CONSTANTS-----------------------------------------------------------------

#DEFINE FUNCTIONS--------------------------------------------------------------
def get_max_counts(temp_df,uni_stns):
    max_counts = {}
    for stn in uni_stns:
        if stn in INT_EXCEPT.keys():
            max_count = INT_EXCEPT[stn]
        else:
            stn_df = temp_df[temp_df[SRC_KEY]==stn].sort_values(by=SRC_TIME)
            stn_times = pd.to_datetime(stn_df[SRC_TIME])
            stn_ints = (stn_times.round('min').diff().dropna().dt.seconds/60).values
            if len(stn_ints) < 1:
                continue
            vals, counts = np.unique(stn_ints,return_counts=True)
            mode_id = np.argmax(counts)
            mode_val = vals[mode_id]
            max_count = (24*60)/mode_val
        max_counts[stn] = max_count
    return max_counts

def get_tmin_tmax(temp_df):
    uni_stns = temp_df[SRC_KEY].unique()
    max_counts = get_max_counts(temp_df,uni_stns)
    temp_data = []
    for stn in uni_stns:
        stn_df = temp_df[temp_df[SRC_KEY]==stn].sort_values(by=SRC_TIME)
        st_date = stn_df[SRC_TIME].values[0]
        date_str = pd.to_datetime(st_date).strftime('X%Y.%m.%d')
        if stn in max_counts.keys():
            stn_max = max_counts[stn]
            stn_counts = stn_df[~stn_df['value'].isna()].drop_duplicates(subset=[SRC_TIME]).shape[0]
            valid_pct = stn_counts/stn_max
            tmin = stn_df[~stn_df['value'].isna()]['value'].min()
            tmax = stn_df[~stn_df['value'].isna()]['value'].max()
            if tmin<tmax:
                temp_data.append([stn,'Tmin',date_str,tmin,valid_pct])
                temp_data.append([stn,'Tmax',date_str,tmax,valid_pct])
    min_max_df = pd.DataFrame(temp_data,columns=[MASTER_KEY,'var','date','value','percent_valid'])
    
    return min_max_df

def convert_dataframe(long_df,varname):
    var_df = long_df[long_df['var']==varname]
    valid_df = var_df[var_df['percent_valid']>=0.95]

    wide_df = pd.DataFrame(index=valid_df[MASTER_KEY].values)
    for stn in wide_df.index.values:
        stn_temp = valid_df[valid_df[MASTER_KEY]==stn].set_index('date')[['value']]
        wide_df.loc[stn,stn_temp.index.values] = stn_temp['value']
    
    wide_df.index.name = MASTER_KEY
    wide_df = wide_df.reset_index()
    return wide_df

""" def update_csv(csv_name,new_data_df):
    master_df = pd.read_csv(MASTER_LINK)
    merged_new_df = master_df.merge(new_data_df,on=MASTER_KEY,how='inner')
    merged_new_df = merged_new_df.set_index('SKN')
    meta_cols = list(master_df.columns)
    if exists(csv_name):
        old_df = pd.read_csv(csv_name)
        old_df = old_df.set_index('SKN')
        updated_df = old_df.merge(merged_new_df,on=meta_cols,how='outer')
        updated_df = updated_df.fillna('NA')
        updated_df = updated_df.reset_index()
        updated_df.to_csv(csv_name,index=False)
    else:
        merged_new_df = merged_new_df.fillna('NA')
        merged_new_df = merged_new_df.reset_index()
        merged_new_df.to_csv(csv_name,index=False) """

def update_csv(csv_name,new_data_df):
    master_df = pd.read_csv(MASTER_LINK)
    prev_ids = new_data_df[MASTER_KEY].values
    merged_new_df = master_df.merge(new_data_df,on=MASTER_KEY,how='inner')
    merged_new_df = merged_new_df.set_index('SKN')
    merged_ids = merged_new_df[MASTER_KEY].values
    unkn_ids = np.setdiff1d(prev_ids,merged_ids)
    master_df = master_df.set_index('SKN')
    meta_cols = list(master_df.columns)
    if exists(csv_name):
        old_df = pd.read_csv(csv_name)
        old_df = old_df.set_index('SKN')
        old_cols = list(old_df.columns)
        old_inds = old_df.index.values
        upd_inds = np.union1d(old_inds,merged_new_df.index.values)
        updated_df = pd.DataFrame(index=upd_inds)
        updated_df.index.name = 'SKN'
        updated_df.loc[old_inds,old_cols] = old_df
        updated_df.loc[merged_new_df.index.values,merged_new_df.columns] = merged_new_df
        updated_df = sort_dates(updated_df,meta_cols)
        updated_df = updated_df.fillna('NA')
        updated_df = updated_df.reset_index()
        updated_df.to_csv(csv_name,index=False)
    else:
        merged_new_df = merged_new_df.fillna('NA')
        merged_new_df = merged_new_df.reset_index()
        merged_new_df.to_csv(csv_name,index=False)
    
    return unkn_ids

def sort_dates(df,meta_cols):
    non_meta_cols = [col for col in list(df.columns) if col not in meta_cols]
    date_keys_sorted = sorted(pd.to_datetime([dt.split('X')[1] for dt in non_meta_cols]))
    date_cols_sorted = [dt.strftime('X%Y.%m.%d') for dt in date_keys_sorted]
    sorted_cols = meta_cols + date_cols_sorted
    sorted_df = df[sorted_cols]
    return sorted_df

def update_unknown(unknown_file,unknown_ids,date_str):
    if exists(unknown_file):
        prev_df = pd.read_csv(unknown_file)
        preex_ids = np.intersect1d(unknown_ids,list(prev_df['sourceID'].values))
        new_ids = np.setdiff1d(unknown_ids,list(prev_df['sourceID'].values))
        prev_df = prev_df.set_index('sourceID')
        prev_df.loc[preex_ids,'lastDate'] = date_str
        prev_df = prev_df.reset_index()
        data_table = [[new_ids[i],SOURCE,date_str] for i in range(len(new_ids))]
        unknown_df = pd.DataFrame(data_table,columns=['sourceID','datastream','lastDate'])
        prev_df = pd.concat([prev_df,unknown_df],axis=0,ignore_index=True)
        prev_df.to_csv(unknown_file,index=False)
    else:
        data_table = [[unknown_ids[i],SOURCE,date_str] for i in range(len(unknown_ids))]
        unknown_df = pd.DataFrame(data_table,columns=['sourceID','datastream','lastDate'])
        unknown_df.to_csv(unknown_file,index=False)

def get_station_sorted_temp(datadir,date_str,output_dir):
    date_dt = pd.to_datetime(date_str)
    date_year = date_dt.strftime('%Y')
    date_month = date_dt.strftime('%m')
    src_file = datadir + '_'.join((date_dt.strftime('%Y%m%d'),SOURCE,'parsed.csv'))
    src_df = pd.read_csv(src_file)
    

    full_temp_df = src_df[src_df[SRC_VARKEY]==SRC_VARNAME]
    hfmet_temp = full_temp_df[full_temp_df['source']=='hfmetar']
    meso_temp = full_temp_df[full_temp_df['source']=='mesonet']

    #Get minmax dataframe long format
    hfmet_minmax = get_tmin_tmax(hfmet_temp)
    meso_minmax = get_tmin_tmax(meso_temp)
    all_minmax = pd.concat([hfmet_minmax,meso_minmax],axis=0,ignore_index=True)

    wide_tmin = convert_dataframe(all_minmax,'Tmin')
    wide_tmax = convert_dataframe(all_minmax,'Tmax')

    #Create new processed table if none exists
    #Update previous with new daily data otherwise
    tmin_process_file = output_dir + '_'.join(('Tmin',SOURCE,date_year,date_month,'processed.csv'))
    tmax_process_file = output_dir + '_'.join(('Tmax',SOURCE,date_year,date_month,'processed.csv'))
    
    tmin_unkn = update_csv(tmin_process_file,wide_tmin)
    tmax_unkn = update_csv(tmax_process_file,wide_tmax)

    tmin_unknown_file = TRACK_DIR + '_'.join(('unknown_Tmin_sta',date_year,date_month)) + '.csv'
    tmax_unknown_file = TRACK_DIR + '_'.join(('unknown_Tmax_sta',date_year,date_month)) + '.csv'

    update_unknown(tmin_unknown_file,tmin_unkn,date_str)
    update_unknown(tmax_unknown_file,tmax_unkn,date_str)


        
#END FUNCTIONS-----------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        hst = pytz.timezone('HST')
        today = datetime.today().astimezone(hst)
        prev_day = today - timedelta(days=1)
        date_str = prev_day.strftime('%Y-%m-%d')

    get_station_sorted_temp(SOURCE_DIR,date_str,PROC_OUTPUT_DIR)
    