import sys
import pytz
import numpy as np
import pandas as pd
from os.path import exists
from datetime import datetime,timedelta

MASTER_DIR = '/home/hawaii_climate_products_container/preliminary/'
DATA_AQS_DIR = MASTER_DIR + 'data_aqs/data_outputs/'
DATA_DIR = DATA_AQS_DIR + 'hi_mesonet/parse/'
PROC_DIR = MASTER_DIR + 'air_temp/working_data/processed_data/hiMeso/'
SOURCE = 'hiMeso'
VARNAME_SET = ['Tair_1_Avg','Tair_2_Avg']
MIN_PER_DAY = 1440
TMIN_NAME = 'Tmin'
TMAX_NAME = 'Tmax'
TIME_COL = 'TIMESTAMP'
REC_COL = 'RECORD'
SKN_CONV = {119:324.6,151:339.4,153:339.6,281:87.9,282:132.2,283:127.5,286:69.25,287:93.13,288:70.7,143:257.11,152:339.5,502:775.11,601:1117.11,602:1132.11}
MASTER_LINK = 'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'
MASTER_DF = pd.read_csv(MASTER_LINK)

def mode(arr):
    vals,counts = np.unique(arr,return_counts=True)
    mode_id = np.argmax(counts)
    mode_val = vals[mode_id]
    return mode_val

def replace_outlier(record_df,stn):
    clim_dir = MASTER_DIR + 'data/clim_atlas/'
    skn = SKN_CONV[stn]
    diffs = record_df.sort_values(['RECORD','TIMESTAMP']).groupby('RECORD')['value'].diff()
    big_diffs = diffs[diffs>5]
    if big_diffs.shape[0]>0:
        recs = record_df.loc[big_diffs.index,'RECORD'].values
        for rec in recs:
            hour = pd.to_datetime(record_df[record_df['RECORD']==rec]['TIMESTAMP'].values[0]).hour
            mon = pd.to_datetime(record_df[record_df['RECORD']==rec]['TIMESTAMP'].values[0]).month
            clim_file = clim_dir + '_'.join(('Tair_clim_atlas',str(hour)))+'.csv'
            clim_df = pd.read_csv(clim_file)
            loni,lati = tuple(MASTER_DF.set_index('SKN').loc[skn,['LON','LAT']])
            min_lon = np.abs(clim_df['LON']-loni).min()
            min_lat = np.abs(clim_df['LAT']-lati).min()
            dist_lon = np.abs(clim_df['LON']-loni)
            dist_lat = np.abs(clim_df['LAT']-lati)
            ind = np.intersect1d(dist_lat[dist_lat==min_lat].index.values,dist_lon[dist_lon==min_lon].index.values)[0]
            clim = clim_df.loc[ind,str(mon)]
            rep_loc = np.abs(record_df[record_df['RECORD']==rec]['value'] - clim).idxmax()
            record_df.loc[rep_loc,'value'] = np.nan
    return record_df

def get_stn_max_count(var_df,stn):
    stn_df = var_df[var_df['sta_ID']==stn].drop_duplicates(subset=[TIME_COL,'var'])
    stn_times = pd.to_datetime(stn_df[TIME_COL].str[:10],format='%Y-%m-%d') + pd.to_timedelta(stn_df[TIME_COL].str[11:16]+':00')
    stn_times = stn_times - pd.Timedelta(1,'s')
    stn_times = stn_times.sort_values()
    stn_ints = (stn_times.round('min').diff().dropna().dt.total_seconds()/60).values
    stn_ints = stn_ints[stn_ints>0]
    mode_int = mode(stn_ints)
    if mode_int <= 60:
        #At most hourly, elsewise, too infrequent reporting
        return MIN_PER_DAY/mode_int
    else:
        return -1.

def convert_wide(data_df,varname):
    daily_var_df = data_df[data_df['var']==varname]
    daily_valid = daily_var_df[daily_var_df['percent_valid']>=0.95]
    daily_match = daily_valid[daily_valid['SKN'].isin(list(SKN_CONV.keys()))].copy()
    daily_match['SKN'] = daily_match['SKN'].replace(SKN_CONV)
    daily_match = daily_match.set_index('SKN')[['value']]
    date_col = daily_valid['date'].unique()[0]
    daily_match = daily_match.rename(columns={'value':date_col})
    daily_meta = MASTER_DF.set_index('SKN').loc[daily_match.index]
    daily_join = daily_meta.join(daily_match,how='right')
    return daily_join

def get_min_max_temp(var_df):
    uni_stns = var_df['sta_ID'].unique()
    min_max_temp = []
    uni_stns = np.intersect1d(uni_stns,list(SKN_CONV.keys()))
    for stn in uni_stns:
        max_count = get_stn_max_count(var_df,stn)
        stn_df = var_df[var_df['sta_ID']==stn]
        record_df = stn_df[['RECORD',TIME_COL,'value']].copy()
        record_df = record_df[~record_df['value'].isna()].copy()
        record_df.loc[:,'value'] = record_df['value'].apply(pd.to_numeric)
        #Placeholder for outlier replacement function before applying the groupby.mean() op
        record_df = replace_outlier(record_df,stn)
        record_df = record_df.groupby(by=['RECORD',TIME_COL]).mean()
        record_df = record_df.reset_index()
        tmin = record_df['value'].min()
        tmax = record_df['value'].max()
        data_count = record_df.shape[0]
        valid_pct = data_count/max_count
        fmt_date_str = pd.to_datetime(record_df[TIME_COL][0].split()[0]).strftime('X%Y.%m.%d')
        min_max_temp.append([stn,'Tmin',fmt_date_str,tmin,valid_pct])
        min_max_temp.append([stn,'Tmax',fmt_date_str,tmax,valid_pct])
    min_max_df = pd.DataFrame(min_max_temp,columns=['SKN','var','date','value','percent_valid'])
    wide_tmin = convert_wide(min_max_df,'Tmin')
    wide_tmax = convert_wide(min_max_df,'Tmax')
    return wide_tmin,wide_tmax

def get_station_sorted_temp(date_str):
    date_dt = pd.to_datetime(date_str)
    year_str = date_dt.strftime('%Y')
    mon_str = date_dt.strftime('%m')
    parsed_file = DATA_DIR + '_'.join((''.join(date_str.split('-')),SOURCE,'parsed')) +'.csv'
    parsed_df = pd.read_csv(parsed_file)
    var_df = parsed_df[parsed_df['var'].isin(VARNAME_SET)]
    wide_tmin,wide_tmax = get_min_max_temp(var_df)
    tmin_process_file = PROC_DIR + '_'.join((TMIN_NAME,SOURCE,year_str,mon_str,'processed'))+'.csv'
    tmax_process_file = PROC_DIR + '_'.join((TMAX_NAME,SOURCE,year_str,mon_str,'processed'))+'.csv'

    update_csv(tmin_process_file,wide_tmin)
    update_csv(tmax_process_file,wide_tmax)

def update_csv(process_file,new_data_df):
    meta_cols = list(MASTER_DF.set_index('SKN').columns)
    if exists(process_file):
        old_df = pd.read_csv(process_file)
        old_df = old_df.set_index('SKN')
        upd_inds = np.union1d(old_df.index.values,new_data_df.index.values)
        updated_df = pd.DataFrame(index=upd_inds)
        updated_df.index.name = 'SKN'
        updated_df.loc[old_df.index.values,old_df.columns] = old_df
        updated_df.loc[new_data_df.index,new_data_df.columns] = new_data_df
        updated_df = sort_dates(updated_df,meta_cols)
        updated_df = updated_df.fillna('NA')
        updated_df = updated_df.reset_index()
        updated_df.to_csv(process_file,index=False)
    else:
        new_data_df = new_data_df.fillna('NA')
        new_data_df = new_data_df.reset_index()
        new_data_df.to_csv(process_file,index=False)

def sort_dates(df,meta_cols):
    non_meta_cols = [col for col in list(df.columns) if col not in meta_cols]
    date_keys_sorted = sorted(pd.to_datetime([dt.split('X')[1] for dt in non_meta_cols]))
    date_cols_sorted = [dt.strftime('X%Y.%m.%d') for dt in date_keys_sorted]
    sorted_cols = meta_cols + date_cols_sorted
    sorted_df = df[sorted_cols]
    return sorted_df

if __name__=="__main__":
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        hst = pytz.timezone('HST')
        today = datetime.today().astimezone(hst)
        prev_day = today - pd.Timedelta(1,'day')
        date_str = prev_day.strftime('%Y-%m-%d')

    get_station_sorted_temp(date_str)