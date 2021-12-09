#!/usr/bin/env python
# coding: utf-8

"""
Version 2.1nrt (NEAR REAL TIME IMPLEMENTATION)
Updated: 12/02/2021

Description
-Refactored from version 1 to accommodate calls from near-real-time processor and wrapper script
-__main__ functionality allows it to be run alone in batch over specified date range to generate maps
-generate_outputs() is the primary function, called from NRT wrapper to generate maps in full workflow
-Updated from previous version to deprecate the metadata call in generate_outputs()
    --A different module will be called to handle output validation

Recent updates:
-Adjusted directorie dependencies to be set exclusively at top level

Pending updates:
-Streamlining helper functions will be necessary for better readability
"""


# In[1]:

import sys
import rasterio
from rasterio.plot import show
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import pandas as pd
from affine import Affine
from pyproj import Transformer

import Temp_linear as tmpl

#DEFINE CONSTANTS-------------------------------------------------------------
MASTER_DIR = r'/home/kodamak8/share/air_temp/'
CODE_MASTER_DIR = MASTER_DIR + r'code/'
WORKING_MASTER_DIR = MASTER_DIR + r'working_data/'
RUN_MASTER_DIR = MASTER_DIR + r'preliminary_output/' #Change to provisional/operational/archival _output as needed
PARAM_TIFF_DIR = WORKING_MASTER_DIR + r'geoTiffs_250m/' #Fixed dir for location of parameter geotiffs
CLIM_FILL_DIR = WORKING_MASTER_DIR + r'clim/'
PRED_DIR = WORKING_MASTER_DIR + r'predictors/'
RAW_DATA_DIR = RUN_MASTER_DIR + r'tables/station_data/daily/raw/' #Location of station and predictor data for model fit
MAP_OUTPUT_DIR = RUN_MASTER_DIR + r'tiffs/county/temp/' #Location of geotiff/png output
SE_OUTPUT_DIR = RUN_MASTER_DIR + r'tiffs/county/temp_SE/'
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

def generate_outputs(iCode, varname, params, date_str, output_dir=None):

    if output_dir == None:
        map_dir = MAP_OUTPUT_DIR + iCode.upper() + '/'
        se_dir = SE_OUTPUT_DIR + iCode.upper() + '/'
        png_dir = PNG_OUTPUT_DIR + iCode.upper() + '/'
    else:
        map_dir = output_dir[0]
        se_dir = output_dir[1]
        png_dir = output_dir[2]

    dateSpl = date_str.split('-')
    dateTail = dateSpl[0] + dateSpl[1] + dateSpl[2]

    #Specify input values
    predname = varname.lower()
    threshold = 2.5     # threshold of sigma clipping when removing outliers
    inversion = 2150    # meter
    #mixHighAlt = 2150   # meter

    #Edit according to master directory standard
    climname = CLIM_FILL_DIR + varname + '_stn_clim.csv'
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
    pred_df,pr_series = tmpl.extract_predictors(pred_filename)
    temp_date = tmpl.get_temperature_date(temp_data,temp_meta,iCode,date_str,varname=varname,climloc=climname)
    df_date = pd.concat([temp_date[[varname,'Island','LON','LAT']],pr_series.loc[temp_date.index.values]],axis=1)

    #Fit temperature model
    temp_series = df_date[varname]
    pr_series = df_date[params]
    MODEL = tmpl.myModel(inversion=inversion)
    theta, pcov, X, y = tmpl.makeModel(temp_series,pr_series,MODEL,threshold=threshold)
    island_df, mask, shape = get_island_grid(iCode, params)
    se_model = tmpl.get_std_error(pr_series,temp_series,pcov,island_df[params],inversion)

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

    cols, rows = shape

    #Generate geotiffs for temperature map and standard error
    if TEMP_TIFF_ON:
        #--Gridded temperature
        fp = genTiffName(param='dem', iCode=iCode)

        ds = gdal.Open(fp)

        # arr_out = np.where((arr < arr_mean), -100000, arr)
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(Tmap_tiff_path, rows, cols, 1, gdal.GDT_Float64)
        # sets same geotransform as input
        outdata.SetGeoTransform(ds.GetGeoTransform())
        outdata.SetProjection(ds.GetProjection())  # sets same projection as input
        outdata.GetRasterBand(1).WriteArray(T_model.reshape(shape))
        # if you want these values (in the mask) transparent
        outdata.GetRasterBand(1).SetNoDataValue(0)
        outdata.FlushCache()  # saves to disk!!
        outdata = None
        ds = None

    #--Gridded standard error
    if SE_TIFF_ON:
        fp = genTiffName(param='dem', iCode=iCode)

        ds = gdal.Open(fp)

        # arr_out = np.where((arr < arr_mean), -100000, arr)
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(se_tiff_path, rows, cols, 1, gdal.GDT_Float64)
        # sets same geotransform as input
        outdata.SetGeoTransform(ds.GetGeoTransform())
        outdata.SetProjection(ds.GetProjection())  # sets same projection as input
        outdata.GetRasterBand(1).WriteArray(se_model.reshape(shape))
        # if you want these values (in the mask) transparent
        outdata.GetRasterBand(1).SetNoDataValue(0)
        outdata.FlushCache()  # saves to disk!!
        outdata = None
        ds = None

    if TEMP_PNG_ON:
        #----------------------------------------------------------------------------
        #PNG generation, might not want to activate this tbh outside of test scenarios.
        # In[34]:

        fig = plt.figure(figsize=(9, 9), dpi=80)
        ax = fig.add_subplot(1, 1, 1)

        fp = Tmap_tiff_path

        raster_img = rasterio.open(fp)

        myarray = raster_img.read(1)
        msk = raster_img.read_masks(1)

        msk[msk > 0] = 1
        image = myarray * msk

        img = ax.imshow(image, cmap='viridis')
        show(raster_img, ax=ax, cmap='viridis')

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

        fig.savefig(Tmap_png_path, dpi=200)
        plt.close("all")

    if SE_PNG_ON:
        fig = plt.figure(figsize=(9, 9), dpi=80)
        ax = fig.add_subplot(1, 1, 1)

        fp = se_tiff_path

        raster_img = rasterio.open(fp)

        myarray = raster_img.read(1)
        msk = raster_img.read_masks(1)

        msk[msk > 0] = 1
        image = myarray * msk

        img = ax.imshow(image, cmap='viridis')
        show(raster_img, ax=ax, cmap='viridis')

        cbar = fig.colorbar(img, ax=ax, shrink=0.6)

        cbar.set_label(
            r'$Standard error \/\/ [^oC]$',
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

        fig.savefig(se_png_path, dpi=200)
        plt.close("all")
    
    return (df_date,theta,isl_dims)

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
        generate_outputs(iCode, varname, params, date_str, outputs)
    else:
    
        for dt in date_list:
            try:
                date_str = str(dt.date())
                print(iCode, varname, date_str)
                generate_outputs(iCode, varname, params, date_str, outputs)
                print('Done -------------^ ')
            except BaseException:
                print('Error -------------^ ')
                #Write error dates to log file
                
