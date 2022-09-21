import sys
import pandas as pd
import numpy as np
from os.path import exists
from datetime import datetime, timedelta
#DEFINE CONSTANTS-------------------------------------------------------------
SRC_DIR = '/home/hawaii_climate_products_container/preliminary/air_temp/data_outputs/tables/station_data/daily/raw_qc/statewide/'
OUTPUT_DIR = '/home/hawaii_climate_products_container/preliminary/air_temp/data_outputs/tables/station_data/monthly/raw_qc/statewide/'
MASTER_META = r'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'
#END CONSTANTS----------------------------------------------------------------

#DEFINE FUNCTIONS-------------------------------------------------------------
def update_df(old_df,new_df):
    old_index = old_df.index.values
    new_index = new_df.index.values
    upd_index = np.union1d(old_index,new_index)
    upd_df = pd.DataFrame(index=upd_index)
    upd_df.loc[old_index,old_df.columns] = old_df.values
    upd_df.loc[new_index,new_df.columns] = new_df.values
    return upd_df

def monthly_mean(temp_data,date_str,threshold=15):
    #Get monthly mean of temperature for specified island
    #Instead of labeling varname, label with month-year
    #Checks which stations are above the data availability threshold
    #Only takes the mean of these stations
    valid_counts = temp_data.count(axis=1)
    stns_threshold = valid_counts[valid_counts>=threshold].index.values
    valid_mean = temp_data.loc[stns_threshold].mean(axis=1).rename(date_str)

    month_df = pd.DataFrame(index=stns_threshold,columns=[date_str])
    month_df.index.name = 'SKN'
    month_df.loc[stns_threshold,date_str] = valid_mean
    month_df[date_str] = pd.to_numeric(month_df[date_str])
    #Only outputs dataframe with SKN index and monthly averaged column
    return month_df

def update_monthly_file(date_id,varname):
    master_df = pd.read_csv(MASTER_META)
    master_df = master_df.set_index('SKN')
    meta_cols = list(master_df.columns)
    year_str = date_id.strftime('%Y')
    mon_str = date_id.strftime('%m')
    filename = OUTPUT_DIR + '_'.join(('monthly',varname,year_str,'qc')) + '.csv'
    this_file = SRC_DIR + '_'.join(('daily',varname,year_str,mon_str,'qc')) + '.csv'
    daily_df = pd.read_csv(this_file)
    daily_df = daily_df.set_index('SKN')
    daily_data = daily_df[[col for col in list(daily_df.columns) if col not in meta_cols]]
    mon_df = monthly_mean(daily_data,'X'+'.'.join((year_str,mon_str)))
    new_meta = master_df.loc[mon_df.index]
    mon_df = new_meta.join(mon_df,how='left')
    if exists(filename):
        old_df = pd.read_csv(filename)
        old_df = old_df.set_index('SKN')
        upd_inds = np.union1d(old_df.index.values,mon_df.index.values)
        #Backfill old
        upd_df = pd.DataFrame(index=upd_inds)
        upd_df.loc[old_df.index,old_df.columns] = old_df.copy()
        #Fill new
        upd_df.loc[mon_df.index,mon_df.columns] = mon_df.copy()
        #Write file
        upd_df = upd_df.reset_index()
        upd_df = upd_df.fillna('NA')
        upd_df.to_csv(filename,index=False)
    else:
        mon_df = mon_df.reset_index()
        mon_df = mon_df.fillna('NA')
        mon_df.to_csv(filename,index=False)
        


#END FUNCTIONS----------------------------------------------------------------

if __name__=="__main__":
    varname = sys.argv[1]
    if len(sys.argv) > 2:
        today_in = sys.argv[2]
        today = pd.to_datetime(today_in)
    else:
        today = datetime.today()
    this_year = today.year
    this_month = today.month
    month_st = datetime(this_year,this_month,1)
    prev = month_st - timedelta(days=1)
    prev_year = prev.year
    prev_month = prev.month
    prev_st = datetime(prev_year,prev_month,1)
    
    #based on what month it is, create previous full month's monthly mean, append to year file
    update_monthly_file(prev_st,varname)





