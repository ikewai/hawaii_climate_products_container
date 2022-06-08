"""
____README____
Version 1.1
updated: 5/26/22 (unmerged)
Description:
-Module file. Can be run as command-line function or aggregate_input can be imported into wrapper.
User specifies source and source file. aggregate_input appends source input data to aggregated
data file (daily_[varname]_YYYY_MM.csv). Overwrites previous data for dates that source data
already exists, otherwise appends as new date data at end of file.
-Creates monthly-separated files with daily data, sorted by SKN.
-If source data file spans the end of the month, creates a new month file and continues appending
to new file.
"""
import sys
import os
import numpy as np
import pandas as pd
from os.path import exists

#DEFINE CONSTANTS--------------------------------------------------------------
#META_COL_N = 12
IDX_NAME = 'SKN'
MASTER_DIR = r'/home/hawaii_climate_products_container/preliminary/'
WORKING_MASTER_DIR = MASTER_DIR + r'air_temp/working_data/'
RUN_MASTER_DIR = MASTER_DIR + r'air_temp/data_outputs/'
PROC_DATA_DIR = WORKING_MASTER_DIR + r'processed_data/'
AGG_OUTPUT_DIR = RUN_MASTER_DIR + r'tables/station_data/daily/raw/statewide/'
META_MASTER_FILE = r'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'
#END CONSTANTS-----------------------------------------------------------------

#DEFINE FUNCTIONS--------------------------------------------------------------
def get_master_meta(master_file=META_MASTER_FILE):
    #Just for encapsulation
    meta_df = pd.read_csv(master_file)
    return meta_df

def update_temp_meta(master_df,temp_df):
    merge_on_cols = list(master_df.columns)
    #Do a col by col check of like columns in temp_df, ensure properly type cast before merge
    #If not same, convert to object type before merge
    temp_df = temp_df.reset_index()
    temp_meta_cols = temp_df.columns[temp_df.columns.isin(list(master_df.columns))]
    for col in temp_meta_cols:
        #if types are same, pass
        #if types not the same and one is object, convert both to object
        if master_df[col].dtypes != temp_df[col].dtypes:
            master_df = master_df.astype({col: 'object'})
            temp_df = temp_df.astype({col: 'object'})
    updated_temp = temp_df.merge(master_df,how='inner',on=merge_on_cols)
    return updated_temp

def update_input_file(df,output_file,master_file=META_MASTER_FILE):
    """
    df: dataframe indexed by SKN or other station index code, column keys are date string only, no meta
    """
    meta_df = pd.read_csv(master_file)
    meta_df = meta_df.set_index(IDX_NAME)
    meta_cols = list(meta_df.columns)
    exist_df = pd.read_csv(output_file)
    exist_df = exist_df.set_index(IDX_NAME)
    exist_cols = exist_df.columns
    data_df = exist_df[[col for col in list(exist_cols) if col not in meta_cols]]
    
    #Update unique indices with union of previous and new
    updated_inds = np.union1d(data_df.index.values,df.index.values)
    updated_df = pd.DataFrame(index=updated_inds)
    updated_df.index.name = IDX_NAME
    #Repopulate new dataframe with old data
    updated_df.loc[data_df.index,data_df.columns] = data_df
    #Check which new data overlaps
    overlap_inds = np.intersect1d(data_df.index.values,df.index.values)
    overlap_cols = [col for col in list(df.columns) if col in list(data_df.columns)]
    overlap_old_arr = data_df.loc[overlap_inds,overlap_cols].values
    overlap_new_arr = df.loc[overlap_inds,overlap_cols].values

    #Replace values where new data is nan with whatever was previously there 
    new_isnan = np.where(np.isnan(overlap_new_arr))
    overlap_new_arr[new_isnan] = overlap_old_arr[new_isnan]
    df.loc[overlap_inds,overlap_cols] = overlap_new_arr
    
    #Add new data where relevant. Overwrite old overlapping data
    updated_df.loc[df.index,df.columns] = df
    
    #date_strs = updated_df.columns.values
    #refrm_dates = ['X'+'.'.join(dt.split('-')) for dt in date_strs]
    #Any date columns that have all missing data are removed prior to merge
    updated_df = updated_df.dropna(how='all')
    #updated_df.columns = refrm_dates
    #Connect to master meta by unique index
    meta_df = meta_df.loc[updated_df.index]
    updated_df = meta_df.join(updated_df,how='left')
    
    return updated_df

def aggregate_input(varname,filename,datadir,outdir,master_file=META_MASTER_FILE):
    master_df = pd.read_csv(master_file)
    master_df = master_df.set_index(IDX_NAME)
    full_filename = datadir + filename
    df = pd.read_csv(full_filename)
    df = df.set_index(IDX_NAME)
    meta_cols = list(master_df.columns)
    temp_cols = [col for col in list(df.columns) if col not in meta_cols]
    meta_df = df[meta_cols]
    temp_df = df[temp_cols]
    
    stn_list = df.index.values
    #Adjust this for Tmin/Tmax_QC.csv. After standardized date format from processing,
    #reset conversion to datetime
    date_keys = pd.to_datetime([x.split('X')[1] for x in list(temp_cols)])
    #date_keys = pd.to_datetime(list(temp_cols))
    st_date = date_keys[0]
    en_date = date_keys[-1]
    st_date_str = ''.join(str(st_date.date()).split('-'))[:-2]
    en_date_str = ''.join(str(en_date.date()).split('-'))[:-2]
    temp_df.columns = date_keys
    if st_date_str == en_date_str:
        #same month
        #Check if file exists. If exists, just update
        #Else, just write the entire dataframe to new file
        year = st_date.year
        mon = st_date.month
        outfile_name = outdir + '_'.join(('daily',varname,str(year),'{:02d}'.format(mon))) + '.csv'
        date_keys_str = [dt.strftime('X%Y.%m.%d') for dt in date_keys]
        temp_df.columns = date_keys_str
        if exists(outfile_name):
            #if file exists, make sure it's not empty
            if os.stat(outfile_name).st_size != 0:
                month_meta = update_input_file(temp_df,outfile_name,master_file)
                month_meta = sort_dates(month_meta,meta_cols)
                month_meta = month_meta.fillna('NA')
                month_meta = month_meta.reset_index()
                month_meta.to_csv(outfile_name,index=False)
            else:
                #if empty, just overwrite with new aggregated file from dataframe
                temp_df = temp_df.dropna(how='all')
                temp_df.columns = temp_cols
                master_df = pd.read_csv(master_file)
                updated_meta = meta_df.loc[temp_df.index]
                full_df = updated_meta.join(temp_df,how='left')
                month_meta = update_temp_meta(master_df,full_df)
                month_meta = month_meta.fillna('NA')
                month_meta.to_csv(outfile_name,index=False)
        else:
            #if does not exist, create new file from dataframe
            temp_df = temp_df.dropna(how='all')
            temp_df.columns = temp_cols
            master_df = pd.read_csv(master_file)
            updated_meta = meta_df.loc[temp_df.index]
            full_df = updated_meta.join(temp_df,how='left')
            month_meta = update_temp_meta(master_df,full_df)
            month_meta = month_meta.fillna('NA')
            month_meta.to_csv(outfile_name,index=False)
    else:
        #split by months
        monyear = np.unique([date.to_period('M') for date in date_keys])
        #print(date_keys)
        for my in monyear:
            mon = my.month
            yr = my.year
            month_keys = [dt for dt in date_keys if ((dt.month == mon) & (dt.year == yr))]
            month_df = temp_df[month_keys]
            month_keys_str = [mk.date().strftime('X%Y.%m.%d') for mk in month_keys]
            month_df.columns = month_keys_str
            outfile_name = outdir + '_'.join(('daily',varname,str(yr),'{:02d}'.format(mon))) + '.csv'
            #check to see if file exists
            if exists(outfile_name):
                if os.stat(outfile_name).st_size != 0:
                    #Updates and overwrites previous version of data file
                    month_meta = update_input_file(month_df,outfile_name,master_file)
                    month_meta = sort_dates(month_meta,meta_cols)
                    month_meta = month_meta.fillna('NA')
                    month_meta = month_meta.reset_index()
                    month_meta.to_csv(outfile_name,index=False)
                else:
                    #create new aggregate file
                    #update this with master
                    month_meta = month_df.dropna(how='all')
                    date_strs = month_meta.columns.values
                    refrm_dates = ['X'+'.'.join(dt.split('-')) for dt in date_strs]
                    month_meta.columns = refrm_dates
                    updated_meta = meta_df.loc[month_meta.index]
                    month_meta = updated_meta.join(month_df,how='left')
                    month_meta = sort_dates(month_meta,meta_cols)
                    month_meta = month_meta.fillna('NA')
                    month_meta = month_meta.reset_index()
                    month_meta.to_csv(outfile_name,index=False)
            else:
                #create new aggregate file
                #update this with master
                month_meta = month_df.dropna(how='all')
                date_strs = month_meta.columns.values
                refrm_dates = ['X'+'.'.join(dt.split('-')) for dt in date_strs]
                month_meta.columns = refrm_dates
                updated_meta = meta_df.loc[month_meta.index]
                month_meta = updated_meta.join(month_df,how='left')
                month_meta = sort_dates(month_meta,meta_cols)
                month_meta = month_meta.fillna('NA')
                month_meta = month_meta.reset_index()
                month_meta.to_csv(outfile_name,index=False)
    
    return month_meta,stn_list
            
def sort_dates(df,meta_cols):
    non_meta_cols = [col for col in list(df.columns) if col not in meta_cols]
    date_keys_sorted = sorted(pd.to_datetime([dt.split('X')[1] for dt in non_meta_cols]))
    date_cols_sorted = [dt.strftime('X%Y.%m.%d') for dt in date_keys_sorted]
    sorted_cols = meta_cols + date_cols_sorted
    sorted_df = df[sorted_cols]
    return sorted_df

def main(varname,filename,datadir,**kwargs):
    varname = str(varname)
    filename = str(filename)
    datadir = str(datadir)
    outdir = None
    master_file = None
    for k,v in kwargs.items():
        if (k=="-mf") | (k=="--master-file"):
            master_file = v
        elif (k == "-o") | (k == "--output-dir"):
            outdir = v
    if master_file == None:
        master_file = META_MASTER_FILE
    if outdir == None:
        outdir = datadir + 'aggregated_input/'
        
    #print(varname,filename,datadir,outdir,master_file)
    return (varname,filename,datadir,outdir,master_file)



#END FUNCTIONS-----------------------------------------------------------------

# __main__
if __name__ == '__main__':
    """
    Arg1: varname
    Arg2: input file name
    Arg3: input source
    Arg4: output location
    Arg5: meta file location
    """
    varname = sys.argv[1]
    filename = sys.argv[2]
    input_source = sys.argv[3]
    
    datadir = PROC_DATA_DIR + input_source + r'/'
    outdir = AGG_OUTPUT_DIR
    master_file = META_MASTER_FILE
    
    print('Outputting',varname,'data from',filename,'to',outdir)
    month_df, stn_list = aggregate_input(varname,filename,datadir,outdir,master_file)