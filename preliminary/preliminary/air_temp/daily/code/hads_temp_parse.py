"""
____README____
VERSION 2.0
BEFORE IMPLEMENT: Please set default system directories/filepaths in CONSTANTS section

Description:
-Command line function: processes HADS data for specified date given as command argument
    -if no argument, default is previous day from today(HST)
-Takes parsed HADS input data and converts to Tmin/max station-sorted time series with meta data
-Should be used in tandem with data aggregator. This is the min/max processing script for HADS.
    -data aggregator combines with other source files into single aggregated file.
-Command line functionality defaults to processing data corresponding to mon-year of current date
-get_station_sorted_temp can be imported as module function and process data for any mon-year spec-
-ified by date_str argument.

Process output in standard file name format:
[varname]_[source]_YYYY_MM_processed.csv
--[source] is 'hads' for HADS processed input stream.

"""
import sys
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from os.path import exists

#DEFINE CONSTANTS--------------------------------------------------------------
SOURCE_NAME = 'hads'
HADS_VARNAME = 'TA'
HADS_VARKEY = 'var'
SRC_KEY = 'staID'
SRC_TIME = 'obs_time'
TMIN_VARNAME = 'Tmin'
TMAX_VARNAME = 'Tmax'
MASTER_KEY = 'NESDIS.id'
MASTER_DIR = r'/home/hawaii_climate_products_container/preliminary/'
SOURCE_DIR = MASTER_DIR + r'data_aqs/data_outputs/' + SOURCE_NAME + r'/parse/'
CODE_MASTER_DIR = MASTER_DIR + r'air_temp/daily/mmcode/'
WORKING_MASTER_DIR = MASTER_DIR + r'air_temp/working_data/'
RUN_MASTER_DIR = MASTER_DIR + r'air_temp/data_outputs/'
PROC_OUTPUT_DIR = WORKING_MASTER_DIR + r'processed_data/' + SOURCE_NAME + r'/'
TRACK_DIR = RUN_MASTER_DIR + r'tables/air_temp_station_tracking/'
MASTER_LINK = r'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'
INT_EXCEPT = {}
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

def get_tmin_tmax(temp_df,date_str):
    uni_stns = temp_df[SRC_KEY].unique()
    max_counts = get_max_counts(temp_df,uni_stns)
    temp_data = []
    obs_time = pd.to_datetime(temp_df[SRC_TIME])
    st_dt = pd.to_datetime(date_str)
    en_dt = st_dt + timedelta(days=1)
    st_dt_utc = st_dt + timedelta(hours=10)
    en_dt_utc = en_dt + timedelta(hours=10)
    obs_times_in_date = obs_time[((obs_time>=st_dt_utc)&(obs_time<en_dt_utc))]
    hst_inds = obs_times_in_date.index.values
    #Hads temperature for date_str date (HST)
    #Duplicate station and time keys dropped
    temp_date = temp_df.loc[hst_inds].drop_duplicates(subset=[SRC_KEY,SRC_TIME])
    date_str = pd.to_datetime(date_str).strftime('X%Y.%m.%d')
    for stn in uni_stns:
        #Temperature data for one specified station, save tmin and tmax
        stn_df = temp_date[temp_date[SRC_KEY]==stn].sort_values(by=SRC_TIME)
        if stn in list(max_counts.keys()):
            stn_max = max_counts[stn]
            stn_counts = stn_df[~stn_df['value'].isna()].shape[0]
            valid_pct = stn_counts/stn_max
            tmin = stn_df[~stn_df['value'].isna()]['value'].min()
            tmax = stn_df[~stn_df['value'].isna()]['value'].max()
            #Convert to deg C
            tmin = (tmin - 32) / 1.8
            tmax = (tmax - 32) / 1.8
            if tmin<tmax:
                temp_data.append([stn,'Tmin',date_str,tmin,valid_pct])
                temp_data.append([stn,'Tmax',date_str,tmax,valid_pct])
            else:
                temp_data.append([stn,'Tmin',date_str,np.nan,valid_pct])
                temp_data.append([stn,'Tmax',date_str,np.nan,valid_pct])

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
        data_table = [[new_ids[i],SOURCE_NAME,date_str] for i in range(len(new_ids))]
        unknown_df = pd.DataFrame(data_table,columns=['sourceID','datastream','lastDate'])
        prev_df = pd.concat([prev_df,unknown_df],axis=0,ignore_index=True)
        prev_df.to_csv(unknown_file,index=False)
    else:
        data_table = [[unknown_ids[i],SOURCE_NAME,date_str] for i in range(len(unknown_ids))]
        unknown_df = pd.DataFrame(data_table,columns=['sourceID','datastream','lastDate'])
        unknown_df.to_csv(unknown_file,index=False)


def get_station_sorted_temp(datadir,date_str,output_dir):
    date_dt = pd.to_datetime(date_str)
    date_year = date_dt.strftime('%Y')
    date_month = date_dt.strftime('%m')
    date_day = date_dt.strftime('%d')
    fname = datadir + '_'.join((''.join(date_str.split('-')),SOURCE_NAME,'parsed')) + '.csv'
    hads_df = pd.read_csv(fname,on_bad_lines='skip',engine='python')

    hads_ta = hads_df[hads_df[HADS_VARKEY] == HADS_VARNAME]
    hads_ta = hads_ta[hads_ta['random']!='R ']

    min_max_df = get_tmin_tmax(hads_ta,date_str)
    wide_tmin = convert_dataframe(min_max_df,'Tmin')
    wide_tmax = convert_dataframe(min_max_df,'Tmax')

    tmin_process_file = output_dir + '_'.join(('Tmin',SOURCE_NAME,date_year,date_month,'processed.csv'))
    tmax_process_file = output_dir + '_'.join(('Tmax',SOURCE_NAME,date_year,date_month,'processed.csv'))

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