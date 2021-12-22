"""
____README____
VERSION 1.1
BEFORE IMPLEMENT: Please set default system directories/filepaths in CONSTANTS section

Description:
-Takes parsed HADS input data and converts to Tmin/max station-sorted time series with meta data
-Should be used in tandem with data aggregator. This is the min/max processing script for HADS.
    -data aggregator combines with other source files into single aggregated file.
-Command line functionality defaults to processing data corresponding to mon-year of current date
-get_station_sorted_temp can be imported as module function and process data for any mon-year spec-
-ified by date_str argument.

Process output in standard file name format:
[varname]_[source]_YYYY_MM_processed_.csv
--[source] is 'hads' for HADS processed input stream.

/home/mplucas/precip_pipeline_container/final_scripts/workflows/dailyDataGet/HADS/outFiles/parse
"""
import sys
import numpy as np
import pandas as pd
from datetime import date

#DEFINE CONSTANTS--------------------------------------------------------------
SOURCE_NAME = 'hads'
HADS_VARNAME = 'TA'
HADS_VARKEY = 'var'
HADS_STNKEY = 'staID'
HADS_TIMEKEY = 'obs_time'
TMIN_VARNAME = 'Tmin'
TMAX_VARNAME = 'Tmax'
FINAL_STNKEY = 'NESDIS.id'
MASTER_DIR = r'/home/hawaii_climate_products_container/preliminary/air_temp/'
SOURCE_DIR = r'/home/mplucas/precip_pipeline_container/final_scripts/workflows/dailyDataGet/' + SOURCE_NAME.upper() + r'/outFiles/parse/'
CODE_MASTER_DIR = MASTER_DIR + r'code/'
WORKING_MASTER_DIR = MASTER_DIR + r'working_data/'
PROC_OUTPUT_DIR = WORKING_MASTER_DIR + r'processed_data/' + SOURCE_NAME + r'/'
META_MASTER_DIR = WORKING_MASTER_DIR + r'static_master_meta/'
META_MASTER_FILE = r'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'
#END CONSTANTS-----------------------------------------------------------------

#DEFINE FUNCTIONS--------------------------------------------------------------
def get_station_interval(var_df):
    uni_stn = var_df[HADS_STNKEY].unique()
    s = []
    t = []
    for stnID in uni_stn:
        stn_per_day = var_df[var_df[HADS_STNKEY]==stnID]
        obs_time = pd.to_datetime(stn_per_day[HADS_TIMEKEY])
        times,counts = np.unique(obs_time.diff().dropna().dt.seconds/60,return_counts=True)
        m_ind = np.argmax(counts)
        t_int = times[m_ind]
        s.append(stnID)
        t.append(t_int)
    station_ints = dict(zip(s,t))
    return station_ints

def get_tminmax_day(var_df,date_str,station_ints):
    obs_time = pd.to_datetime(var_df[HADS_TIMEKEY])
    current_dt = pd.to_datetime(date_str)
    obs_dates = obs_time.dt.date
    var_date = var_df.where(obs_dates==current_dt).dropna().drop_duplicates(subset=[HADS_STNKEY,HADS_TIMEKEY])
    uni_stn = var_date[HADS_STNKEY].unique()
    temp_data = []
    for stnID in uni_stn:
        stn_date = var_date[var_date[HADS_STNKEY]==stnID]
        #print(stn_date.shape)
        #nws_id = var_date[var_date['staID']==stnID]['NWS_sid'].unique()[0]
        max_ct = 24*60/station_ints[stnID]
        tmin = stn_date.dropna()['value'].min()
        tmax = stn_date.dropna()['value'].max()
        #Convert to deg C
        tmin = (tmin - 32) / 1.8
        tmax = (tmax - 32) / 1.8
        valid_pct = stn_date.dropna().shape[0] / max_ct
        temp_data.append([stnID,'Tmin',date_str,tmin,valid_pct])
        temp_data.append([stnID,'Tmax',date_str,tmax,valid_pct])
    min_max_df = pd.DataFrame(temp_data,columns=[FINAL_STNKEY,'var','date','value','percent_valid'])
    return min_max_df

def convert_dataframe(temp_df,stn_indices):
    """
    Converts long-format temperature data to station-indexed, date-column table format
    --temp_df: Expects dataframe indexed by unique station key (e.g. SKN)
    --stn_indices: Unique station keys to set the index of output dataframe
    -
    --temp_wide: Output sorted by station key in rows, with temp data advancing by date in columns
        --date column labels are in format 'X[YYYY].[MM].[DD]' for maximum compatibility
    """
    obs_dates = np.unique(temp_df['date'])
    st = obs_dates[0]
    en = obs_dates[-1]
    st_dt = pd.to_datetime(st)
    en_dt = pd.to_datetime(en)
    date_span = [dt.strftime('%Y-%m-%d') for dt in pd.date_range(st_dt,en_dt)]

    temp_wide = pd.DataFrame(index=stn_indices,columns=date_span)
    for stnID in stn_indices:
        stn_temp = temp_df[temp_df[FINAL_STNKEY]==stnID]
        valid_temp = stn_temp[stn_temp['percent_valid']>=0.95].set_index('date')[['value']]
        temp_wide.loc[stnID,valid_temp.index] = valid_temp['value']
    date_relabel = [dt.strftime('X%Y.%m.%d') for dt in pd.date_range(st_dt,en_dt)]
    temp_wide.columns = date_relabel
    temp_wide.index.name = FINAL_STNKEY
    return temp_wide


def get_station_sorted_temp(datadir,date_str,outdir,master_file=META_MASTER_FILE):
    #Process data
    data_source = SOURCE_NAME
    date_dt = pd.to_datetime(date_str)
    date_year = date_dt.year
    date_month = date_dt.month
    fname = datadir + '_'.join((str(date_year),"{:02d}".format(date_month),SOURCE_NAME,'1am_all_data')) + '.csv'
    hads_df = pd.read_csv(fname,on_bad_lines='skip',engine='python')

    hads_ta = hads_df[hads_df[HADS_VARKEY] == HADS_VARNAME]
    hads_ta = hads_ta[hads_ta['random']!='R ']
    all_times_dt = pd.to_datetime(hads_ta[HADS_TIMEKEY])
    all_dates = all_times_dt.dt.date
    uni_stn = hads_ta[HADS_STNKEY].unique()
    station_intervals = get_station_interval(hads_ta)
    all_tminmax = pd.DataFrame()
    for dt in all_dates.unique():
        date_str = str(dt)
        t_minmax = get_tminmax_day(hads_ta,date_str,station_intervals)
        all_tminmax = pd.concat([all_tminmax,t_minmax],ignore_index=True)
    
    all_tmin = all_tminmax[all_tminmax[HADS_VARKEY]==TMIN_VARNAME]
    all_tmax = all_tminmax[all_tminmax[HADS_VARKEY]==TMAX_VARNAME]

    #Get master meta
    master_df = pd.read_csv(master_file)
    select_meta = master_df[master_df[FINAL_STNKEY].isin(uni_stn)]

    #Station sort Tmin and convert
    tmin_by_stn = convert_dataframe(all_tmin,uni_stn)
    tmin_by_stn = tmin_by_stn.reset_index()
    tmin_by_stn = select_meta.merge(tmin_by_stn,on=FINAL_STNKEY,how='inner')
    tmin_by_stn = tmin_by_stn.fillna('NA')
    tmin_out_file = '_'.join(('Tmin',data_source,str(date_year),str(date_month),'processed')) + '.csv'
    tmin_out = outdir + tmin_out_file
    tmin_by_stn.to_csv(tmin_out,index=False)

    #Station sort Tmax and convert
    tmax_by_stn = convert_dataframe(all_tmax,uni_stn)
    tmax_by_stn = tmax_by_stn.reset_index()
    tmax_by_stn = select_meta.merge(tmax_by_stn,on=FINAL_STNKEY,how='inner')
    tmax_by_stn = tmax_by_stn.fillna('NA')
    tmax_out_file = '_'.join(('Tmax',data_source,str(date_year),str(date_month),'processed')) + '.csv'
    tmax_out = outdir + tmax_out_file
    tmax_by_stn.to_csv(tmax_out,index=False)

    return (outdir,tmin_out_file,tmax_out_file)
    

def main(**kwargs):
    outdir = None
    master_file = None
    for k,v in kwargs.items():
        if (k=="-mf") | (k=="--master-file"):
            master_file = v
        elif (k == "-o") | (k == "--output-dir"):
            outdir = v
    if outdir == None:
        outdir = PROC_OUTPUT_DIR
    if master_file == None:
        master_file = META_MASTER_FILE
    
    return (outdir, master_file)


#END FUNCTIONS-----------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        date_str = date.today().strftime('%Y-%m-%d')

    outdir, master_file = main()
    datadir = SOURCE_DIR
    process_dir,processed_tmin_file, processed_tmax_file = get_station_sorted_temp(datadir,date_str,outdir,master_file)
    
