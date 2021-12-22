#!/usr/bin/env python
# coding: utf-8

"""
Version 2.1nrt (NEAR REAL TIME IMPLEMENTATION)
Updated: 12/02/2021

Description
-Refactored from version 1 to accommodate calls from near-real-time processor and wrapper script
-__main__ functionality allows it to be run alone in batch over specified date range to generate maps
-generate_county_map() is the primary function, called from NRT wrapper to generate maps in full workflow
-Updated from previous version to deprecate the metadata call in generate_county_map()
    --A different module will be called to handle output validation

Recent updates:
-Adjusted directorie dependencies to be set exclusively at top level

Pending updates:
-Streamlining helper functions will be necessary for better readability
"""


# In[1]:

import sys
import rasterio
#from rasterio.plot import show
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import pandas as pd
from os.path import exists
from affine import Affine
from pyproj import Transformer

import Temp_linear as tmpl

#DEFINE CONSTANTS-------------------------------------------------------------
MASTER_DIR = r'/home/hawaii_climate_products_container/preliminary/air_temp/'
CODE_MASTER_DIR = MASTER_DIR + r'code/'
WORKING_MASTER_DIR = MASTER_DIR + r'working_data/'
RUN_MASTER_DIR = MASTER_DIR + r'data_outputs/'
PARAM_TIFF_DIR = WORKING_MASTER_DIR + r'geoTiffs_250m/' #Fixed dir for location of parameter geotiffs
CLIM_FILL_DIR = WORKING_MASTER_DIR + r'clim/'
PRED_DIR = WORKING_MASTER_DIR + r'predictors/'
QC_DATA_DIR = RUN_MASTER_DIR + r'tables/station_data/daily/raw_qc/county/'
RAW_DATA_DIR = RUN_MASTER_DIR + r'tables/station_data/daily/raw/statewide/' #Location of station and predictor data for model fit
MAP_OUTPUT_DIR = RUN_MASTER_DIR + r'tiffs/daily/county/' #Location of geotiff/png output
SE_OUTPUT_DIR = RUN_MASTER_DIR + r'tiffs/daily/county/'
PNG_OUTPUT_DIR = RUN_MASTER_DIR + r'plots/county/maps/'

TEMP_TIFF_ON = True #Set True to output temperature gridded geotiff
SE_TIFF_ON = True   #Set True to output standard error gridded geotiff
TEMP_PNG_ON = False  #Set True to output png display of temperature geotiff
SE_PNG_ON = False    #Set True to output png display of standard error geotiff

#END CONSTANTS----------------------------------------------------------------

#DEFINE FUNCTIONS-------------------------------------------------------------
def get_Coordinates(GeoTiff_name):

    # Read raster
    with rasterio.open(GeoTiff_name) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        A = r.read()  # pixel values

    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing
    # at centre
    def rc2en(r, c): return T1 * (c, r)

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(
        rc2en, otypes=[
            float, float])(
        rows, cols)

    transformer = Transformer.from_proj(
        'EPSG:4326',
        '+proj=longlat +datum=WGS84 +no_defs +type=crs',
        always_xy=True,
        skip_equivalent=True)

    LON, LAT = transformer.transform(eastings, northings)
    return LON, LAT


def genParamCode(param):

    if param == 'coastDist':
        return 'coastDistM'
    elif param == 'rf':
        return 'meanAnn'
    else:
        return param


def genTiffName(param='dem', iCode='bi'):

    iCode = iCode.lower()
    if param == 'dem_250':
        param = 'dem'

    pCODE = genParamCode(param)
    TiffName = PARAM_TIFF_DIR + param + \
        '/' + iCode + '_' + pCODE + '_250m.tif'

    return TiffName


def getDataArray(param='dem_250', iCode='bi', getMask=False):

    TiffName = genTiffName(param=param, iCode=iCode)
    raster_img = rasterio.open(TiffName)

    myarray = raster_img.read(1)
    msk = raster_img.read_masks(1)

    msk[msk > 0] = 1
    dataArray = myarray * msk

    dataArray[msk == 0] = 0

    if getMask:
        return msk, myarray.shape    # 0:reject  >0:accept

    return dataArray

def get_island_grid(iCode, params):

    TiffName = genTiffName(iCode=iCode)
    LON, LAT = get_Coordinates(TiffName)
    LON = LON.reshape(-1)
    LAT = LAT.reshape(-1)

    myDict = {'LON': LON, 'LAT': LAT}

    for param in params:

        myDict[param] = getDataArray(iCode=iCode, param=param).reshape(-1)

    island_df = pd.DataFrame.from_dict(myDict)

    mask, shape = getDataArray(iCode=iCode, getMask=True)

    return island_df, mask.reshape(-1), shape

def G_islandName(iCode):

    if iCode == 'bi':
        return "Big Island"
    elif iCode == 'oa':
        return "Oahu"
    elif iCode == 'mn':
        return "Maui nui"
    elif iCode == 'ka':
        return "Kauai"
    else:
        return iCode

def output_tiff(df_data,tiff_filename,iCode,shape):
    cols, rows = shape
    fp = genTiffName(param='dem', iCode=iCode)
    
    ds = gdal.Open(fp)

    # arr_out = np.where((arr < arr_mean), -100000, arr)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(tiff_filename, rows, cols, 1, gdal.GDT_Float64)
    # sets same geotransform as input
    outdata.SetGeoTransform(ds.GetGeoTransform())
    outdata.SetProjection(ds.GetProjection())  # sets same projection as input
    outdata.GetRasterBand(1).WriteArray(df_data.reshape(shape))
    # if you want these values (in the mask) transparent
    outdata.GetRasterBand(1).SetNoDataValue(0)
    outdata.FlushCache()  # saves to disk!!
    outdata = None
    ds = None

def output_png(tiff_file,png_file,iCode):
    fig = plt.figure(figsize=(9, 9), dpi=80)
    ax = fig.add_subplot(1, 1, 1)

    fp = tiff_file

    raster_img = rasterio.open(fp)

    myarray = raster_img.read(1)
    msk = raster_img.read_masks(1)

    msk[msk > 0] = 1
    image = myarray * msk

    img = ax.imshow(image, cmap='viridis')
    
    cbar = fig.colorbar(img, ax=ax, shrink=0.6)

    cbar.set_label(
        r'$Temperature \/\/ [^oC]$',
        rotation=90,
        fontsize=14,
        labelpad=10)
    cbar.ax.tick_params(labelsize=12)

    # set_axes(ax, (21.2, 21.7), (-158.3, -157.))

    ax.set_xlabel("Longitude [deg]", fontsize=14)
    ax.set_ylabel("Latitude [deg]", fontsize=14)
    # ax.set_title("Big Island (LAI - 250 m)", fontsize=16)

    fontsize = 13

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    ax.set_title(G_islandName(iCode) + " (" + date_str + ")", fontsize=14)

    fig.savefig(png_file, dpi=200)
    plt.close("all")    

def generate_county_map(iCode, varname, params, date_str, output_dir=None):

    if output_dir == None:
        map_dir = MAP_OUTPUT_DIR + varname + '/' + iCode.upper() + '/'
        se_dir = SE_OUTPUT_DIR + varname + '_se/' + iCode.upper() + '/'
        png_dir = PNG_OUTPUT_DIR + iCode.upper() + '/'
    else:
        map_dir = output_dir[0]
        se_dir = output_dir[1]
        png_dir = output_dir[2]

    dateSpl = date_str.split('-')
    dateTail = dateSpl[0] + dateSpl[1] + dateSpl[2]

    #Specify input values
    if iCode == 'MN':
        isl_list = ['MA','KO','MO','LA']
    else:
        isl_list = [iCode]

    predname = varname.lower()
    threshold = 2.5     # threshold of sigma clipping when removing outliers
    inversion = 2150    # meter
    #mixHighAlt = 2150   # meter

    #Edit according to master directory standard
    climname = CLIM_FILL_DIR + varname + '_stn_clim.csv'
    qc_temp_file = QC_DATA_DIR + '_'.join(('daily',varname,iCode,dateSpl[0],dateSpl[1],'qc')) + '.csv'
    Tmap_tiff = varname + '_map_' + iCode + '_' + dateTail + '.tif'
    Tmap_tiff_path = map_dir + Tmap_tiff
    se_tiff = varname + '_map_' + iCode + '_' + dateTail + '_se.tif'
    se_tiff_path = se_dir + se_tiff
    Tmap_png = varname + '_map_' + iCode + '_' + dateTail + '.png'
    Tmap_png_path = png_dir + Tmap_png
    se_png = varname + '_map_' + iCode + '_' + dateTail + '_se.png'
    se_png_path = png_dir + se_png

    #Extract and process data for model fitting
    date_dt = pd.to_datetime(date_str)
    date_month = "{:02d}".format(date_dt.month)
    date_year = str(date_dt.year)
    temp_filename = RAW_DATA_DIR + '_'.join(('daily',varname,date_year,date_month)) + '.csv'
    pred_filename = PRED_DIR + predname + '_predictors.csv'
    temp_df,temp_meta,temp_data = tmpl.extract_temp_input(temp_filename)
    pred_df,pr_series = tmpl.extract_predictors(pred_filename,params)
    temp_date = tmpl.get_temperature_date(temp_data,temp_meta,iCode,date_str,varname=varname,climloc=climname)
    valid_skns = np.intersect1d(temp_date.index.values,pr_series.index.values)
    df_date = pd.concat([temp_date.loc[valid_skns,[varname,'Island','LON','LAT']],pr_series.loc[valid_skns]],axis=1)

    #Fit temperature model
    temp_series = df_date[varname]
    pr_series = df_date[params]
    MODEL = tmpl.myModel(inversion=inversion)
    theta, pcov, X, y = tmpl.makeModel(temp_series,pr_series,MODEL,threshold=threshold)
    island_df, mask, shape = get_island_grid(iCode, params)
    se_model = tmpl.get_std_error(pr_series,temp_series,pcov,island_df[params],inversion)

    #QC output
    today_dt = pd.to_datetime(date_str)
    indx = tmpl.removeOutlier(pr_series.values,temp_series.values,threshold=3)
    #Index of temp_series values (mixed-islands) that are not flagged
    df_indx = temp_series.index.values[indx]
    #Get temp_date non-flagged values 
    unflagged_temp = temp_date.loc[df_indx]
    #Get only target island flagged
    target_isl_qc = unflagged_temp[unflagged_temp['Island'].isin(isl_list)]

    if exists(qc_temp_file):
        temp_qc_prev = pd.read_csv(qc_temp_file)
        temp_qc_prev = temp_qc_prev.set_index('SKN')
        meta_cols = list(temp_meta.columns)
        prev_meta = temp_qc_prev[meta_cols]
        prev_data_cols = [col for col in list(temp_qc_prev.columns) if col not in meta_cols]
        dt_cols = pd.to_datetime([col.split('X')[1] for col in prev_data_cols])
        temp_qc_prev.rename(columns=dict(zip(prev_data_cols,dt_cols)),inplace=True)
        prev_inds = temp_qc_prev.index.values
        new_inds = np.union1d(prev_inds,df_indx)
        new_temp_qc = pd.DataFrame(index=new_inds)
        new_temp_qc.index.name = 'SKN'
        #Backfill prior
        new_temp_qc.loc[temp_qc_prev.index,temp_qc_prev.columns] = temp_qc_prev
        #Add new flagged data in specified column.
        #Dates are formatted as pd.datetime objects
        new_temp_qc.loc[target_isl_qc.index,today_dt] = target_isl_qc[varname].values
        #Convert times
        data_cols = [col for col in list(new_temp_qc.columns) if col not in meta_cols]
        new_dates = [dt.strftime('X%Y.%m.%d') for dt in data_cols]
        col_rename = dict(zip(data_cols,new_dates))
        new_temp_qc.rename(columns=col_rename,inplace=True)
        #Write updated qc table
        new_temp_qc = new_temp_qc.reset_index()
        new_temp_qc.to_csv(qc_temp_file,index=False)
    else:
        #If no prior version of qc_temp_file exists, write new
        #Convert times first
        x_date_str = today_dt.strftime('X%Y.%m.%d')
        target_isl_qc.rename(columns={varname:x_date_str},inplace=True)
        target_isl_qc = target_isl_qc.reset_index()
        target_isl_qc.to_csv(qc_temp_file,index=False)
    
    #Set island dims
    lons = island_df['LON'].unique()
    lats = island_df['LAT'].unique()
    xresolution = np.round(lons[1] - lons[0],6)
    yresolution = np.round(lats[1] - lats[0],6)
    xmin = np.min(lons)
    xmax = np.max(lons)
    ymin = np.min(lats)
    ymax = np.max(lats)
    isl_dims = {'XResolution':xresolution,'YResolution':yresolution,'Xmin':xmin,'Xmax':xmax,'Ymin':ymin,'Ymax':ymax}


    #Do cross-validation output for the given date
    
    #Create geotiff for temperature map and standard error map
    X_island = island_df.values
    T_model = MODEL(X_island[:, 2:], *theta)
    T_model[mask == 0] = np.nan
    se_model[mask == 0] = np.nan

    #Generate geotiffs for temperature map and standard error
    if TEMP_TIFF_ON:
        output_tiff(T_model,Tmap_tiff_path,iCode,shape)

    #--Gridded standard error
    if SE_TIFF_ON:
        output_tiff(se_model,se_tiff_path,iCode,shape)

    if TEMP_PNG_ON:
        output_png(Tmap_tiff_path,Tmap_png_path,iCode)

    if SE_PNG_ON:
        output_png(se_tiff_path,se_png_path,iCode)
    
    return (temp_date,pr_series,theta,isl_dims)

# This code has been automatically covnerted to comply with the pep8 convention
# This the Linux command:
# $ autopep8 --in-place --aggressive  <filename>.py
if __name__ == '__main__':

    iCode = str(sys.argv[1])  # 'bi'
    varname = str(sys.argv[2]) #variable name
    date_range = str(sys.argv[3]) #YYYYMMDD_start-YYYYMMDD_end
    date_range = date_range.split('-')
    st_date = pd.to_datetime(date_range[0],format='%Y%m%d')
    en_date = pd.to_datetime(date_range[1],format='%Y%m%d')

    date_list = pd.date_range(st_date,en_date)

    params = ["dem_250"]

    #All directories set from CONSTANTS
    outputs = None
    iCode = iCode.upper()

    error_on = False
    #Error checking (for testing purposes only)
    if error_on:
        date_str = str(date_list[0].date())
        print(iCode, varname, date_str)
        generate_county_map(iCode, varname, params, date_str, outputs)
    else:
    
        for dt in date_list:
            try:
                date_str = str(dt.date())
                print(iCode, varname, date_str)
                generate_county_map(iCode, varname, params, date_str, outputs)
                print('Done -------------^ ')
            except BaseException:
                print('Error -------------^ ')
                #Write error dates to log file
                
