import sys
import subprocess
import rasterio
import numpy as np
import pandas as pd

import Temp_linear as tmpl
from osgeo import gdal
from affine import Affine
from pyproj import Transformer
from datetime import date
#DEFINE CONSTANTS--------------------------------------------------------------
MASTER_DIR = '/home/hawaii_climate_products_container/preliminary/'
RUN_MASTER_DIR = MASTER_DIR + 'air_temp/data_outputs/'
COUNTY_MAP_DIR = RUN_MASTER_DIR + 'tiffs/monthly/county/'
STATE_MAP_DIR = RUN_MASTER_DIR + 'tiffs/monthly/statewide/'
LOOCV_COUNTY_DIR = RUN_MASTER_DIR + 'tables/loocv/monthly/county/'
LOOCV_STATE_DIR = RUN_MASTER_DIR + 'tables/loocv/monthly/statewide/'
META_OUTPUT_DIR = RUN_MASTER_DIR + 'metadata/monthly/statewide/'
NO_DATA_VAL = -9999
#END CONSTANTS-----------------------------------------------------------------

#DEFINE FUNCTIONS--------------------------------------------------------------
def get_grid_pix(varname,icode_list,date_str):
    year = date_str.split('-')[0]
    mon = date_str.split('-')[1]
    date_tail = year + mon
    input_dir = COUNTY_MAP_DIR + varname + '/'
    file_names = [input_dir+icode.upper()+'/'+'_'.join((varname,'map',icode.upper(),date_tail,'monthly')) + '.tif' for icode in icode_list]

    grid_pix = []
    for f in file_names:
        raster_img = rasterio.open(f)
        raster_data = raster_img.read(1)
        raster_mask = raster_img.read_masks(1)
        raster_mask[raster_mask > 0] = 1
        masked_data = raster_data * raster_mask
        masked_data[raster_mask == 0] = np.nan

        isl_pix = np.sum(~np.isnan(masked_data))
        grid_pix.append(isl_pix)
    
    return grid_pix

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

def output_tiff(data,base_tiff_name,out_tiff_name,tiff_shape):
    cols,rows = tiff_shape
    ds = gdal.Open(base_tiff_name)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(out_tiff_name, rows, cols, 1, gdal.GDT_Float32)
    # sets same geotransform as input
    outdata.SetGeoTransform(ds.GetGeoTransform())
    outdata.SetProjection(ds.GetProjection())  # sets same projection as input
    outdata.GetRasterBand(1).WriteArray(data.reshape(tiff_shape))
    # if you want these values (in the mask) transparent
    outdata.GetRasterBand(1).SetNoDataValue(0)
    outdata.FlushCache()  # saves to disk!!
    outdata = None
    band = None
    ds = None
    
def reformat_tiff(tiffname):
    ra = rasterio.open(tiffname)
    shape = ra.read(1).shape
    
    data = ra.read(1)
    mask = ra.read_masks(1)

    mask[mask > 0] = 1
    masked_data = data * mask
    masked_data[mask==0] = NO_DATA_VAL

    output_tiff(masked_data,tiffname,tiffname,shape)

def statewide_mosaic(varname,date_str,temp_suffix=''):
    icode_list = ['bi','ka','mn','oa']
    year = date_str.split('-')[0]
    mon = date_str.split('-')[1]
    date_tail = year + mon
    input_dir = COUNTY_MAP_DIR + varname + temp_suffix +'/'
    output_dir = STATE_MAP_DIR + varname + temp_suffix + '/'
    file_names = [input_dir+icode.upper()+'/'+'_'.join((varname,'map',icode.upper(),date_tail,'monthly')) + temp_suffix +'.tif' for icode in icode_list]
    output_name = output_dir + '_'.join((varname,'map','state',date_tail,'monthly')) + temp_suffix +'.tif'
    cmd = "gdal_merge.py -o "+output_name+" -of gtiff -co COMPRESS=LZW -n -9999 -a_nodata -9999"
    subprocess.call(cmd.split()+file_names)
    #reformat_tiff(output_name)

def loocv_aggregate(varname,date_str,n_params=1):
    #n_params defaults to elevation-only model. Increase if re-running for more predictors
    icode_list = ['bi','ka','mn','oa']
    year = date_str.split('-')[0]
    mon = date_str.split('-')[1]
    date_tail = year + mon
    file_names = [LOOCV_COUNTY_DIR + icode.upper() + '/' + '_'.join((date_tail,varname,icode.upper(),'loocv_monthly.csv')) for icode in icode_list]
    output_name = LOOCV_STATE_DIR + '_'.join((date_tail,varname,'state','loocv_monthly.csv'))
    #Open each csv, pull relevant data from csv, add to list, concatenate
    df_list = []
    meta = {'StationCountCounties':[],'mae':[],'rmse':[],'r2':[],'aic':[],'aicc':[],'bic':[],'bias':[]}
    for f in file_names:
        df = pd.read_csv(f)
        valid_df = df[df['ValidatedStation']== True]
        i = file_names.index(f)
        isl_loc = np.where(valid_df.columns.values == 'Island')[0][0]
        isl_loc = int(isl_loc) + 1
        #valid_df.insert(loc=isl_loc,column='county',value=counties)
        df_list.append(valid_df)
        predicted = valid_df['PredictedTemp'].values.flatten()
        observed = valid_df['ObservedTemp'].values.flatten()
        pred_clip,obs_clip = tmpl.sigma_Clip(predicted,observed)
        station_count = valid_df.shape[0]
        if ((len(pred_clip) - n_params - 1) < 3) | ((len(obs_clip) - n_params - 1) < 3):
            mae = np.nan
            rmse = np.nan
            r2 = np.nan
            aic = np.nan
            aicc = np.nan
            bic = np.nan
            bias = np.nan
        else:
            mae,rmse,r2,aic,aicc,bic = tmpl.metrics(pred_clip,obs_clip,False,n_params)
            obs_mean = np.mean(obs_clip)
            pred_mean = np.mean(pred_clip)
            bias = obs_mean - pred_mean
        meta['mae'].append(mae)
        meta['rmse'].append(rmse)
        meta['r2'].append(r2)
        meta['aic'].append(aic)
        meta['aicc'].append(aicc)
        meta['bic'].append(bic)
        meta['bias'].append(bias)
        meta['StationCountCounties'].append(station_count)
    all_df = pd.concat(df_list,axis=0,ignore_index=True)
    all_df.drop('ValidatedStation',axis=1,inplace=True)
    pred_all = all_df['PredictedTemp'].values.flatten()
    obs_all = all_df['ObservedTemp'].values.flatten()
    pred_clip_all,obs_clip_all = tmpl.sigma_Clip(pred_all,obs_all)
    if ((len(pred_clip_all) - n_params - 1) <3) | ((len(obs_clip_all) - n_params - 1) < 3):
        mae = np.nan
        rmse = np.nan
        r2 = np.nan
        aic = np.nan
        aicc = np.nan
        bic = np.nan
        bias = np.nan
    else:
        mae,rmse,r2,aic,aicc,bic = tmpl.metrics(pred_clip_all,obs_clip_all,False,n_params)
        obs_mean = np.mean(obs_clip_all)
        pred_mean = np.mean(pred_clip_all)
        bias = obs_mean - pred_mean

    meta['mae'].insert(0,mae)
    meta['rmse'].insert(0,rmse)
    meta['r2'].insert(0,r2)
    meta['aic'].insert(0,aic)
    meta['aicc'].insert(0,aicc)
    meta['bic'].insert(0,bic)
    meta['bias'].insert(0,bias)
    
    all_df = all_df.fillna('NA')
    all_df.to_csv(output_name,index=False)

    return meta

def create_tables(variable,mode,date_str):
    #master_dir is the parent directory above the loocv and metadata folders
    #assumes directory structure obeys standard set by dir_maker.py
    #tiff_dir = master_dir + 'tiffs/'
    #tables_dir = master_dir + 'tables/'
    varname = variable + mode
    meta = loocv_aggregate(varname,date_str)
    fmt_date = pd.to_datetime(date_str).strftime('%b. %Y')
    today = date.today().strftime('%Y-%m-%d')
    year = date_str.split('-')[0]
    mon = date_str.split('-')[1]
    date_tail = year + mon
    min_max = {'min':'minimum','max':'maximum','mean':'average','MIN':'minimum','MAX':'maximum','MEAN':'average'}
    icode_list = ['bi','ka','mn','oa']
    #county_full = {'bi':'Hawaii','ka':'Kauai','mn':'Maui (Maui, Lanai, Molokai, & Kahoolawe)','oa':'Honolulu (Oahu)'}
    station_total = np.sum(meta['StationCountCounties'])
    keyword_val = 'Hawaii, Hawaiian Islands, Temperature prediction, monthly Temperature, Temperature, Climate, Linear Regression'
    if mode == 'mean':
        temp_station_name = '_'.join(('daily','Tmin',year,mon)) \
                            + '.csv, ' + '_'.join(('daily','Tmax',year,mon)) + '.csv'
    else:
        temp_station_name = '_'.join(('daily',varname,year,mon)) + '.csv'
    temp_tiff_name = '_'.join((varname,'map','state',date_tail,'monthly')) + '.tif'
    temp_se_name = '_'.join((varname,'map','state',date_tail,'monthly','se')) + '.tif'

    meta_file = META_OUTPUT_DIR + '_'.join((date_tail,varname,'state','meta','monthly')) + '.txt'

    if meta['r2'][0] >= 0.75:
        quality = 'high quality'
    elif ((meta['r2'][0] < 0.75) & (meta['r2'][0] >= 0.5)):
        quality = 'moderate quality'
    else:
        quality = 'low quality'

    current_tiff = STATE_MAP_DIR + varname + '/' + temp_tiff_name
    LON,LAT = get_Coordinates(current_tiff)
    lons = np.unique(LON)
    lats = np.unique(LAT)
    xmin = str(np.round(np.min(lons),3))
    xmax = str(np.round(np.max(lons),3))
    ymin = str(np.round(np.min(lats),3))
    ymax = str(np.round(np.max(lats),3))
    xres = str(np.round(lons[1] - lons[0],6))
    yres = str(np.round(lats[1] - lats[0],6))

    grid_pix = get_grid_pix(varname,icode_list,date_str)
    grid_pix_total = np.sum(grid_pix)
    grid_pix = ', '.join([str(pix) for pix in grid_pix])
    credit_statement = 'All data produced by University of Hawaii at Manoa Dept. of Geography and the Enviroment, Ecohydology Lab in collaboration with the Water Resource Research Center (WRRC). Support for the Hawai‘i EPSCoR Program is provided by the Hawaii Emergency Management Agency.'
    contact_list = 'Keri Kodama (kodamak8@hawaii.edu), Matthew Lucas (mplucas@hawaii.edu), Ryan Longman (rlongman@hawaii.edu), Sayed Bateni (smbateni@hawaii.edu), Thomas Giambelluca (thomas@hawaii.edu)'
    
    #Data statement
    dataStatement_val = 'This {date} mosaic temperature map of the State of Hawaii is a high spatial resolution (~250m) gridded prediction of {minmax} temperature in degrees Celsius. This was produced using a piece-wise linear regression model using elevation (m) as its predictor(s). This process was done for four individually produced maps of Kauai, Honolulu (Oahu), Maui (Maui, Lanai, Molokai, & Kahoolawe) and Hawaii counties. The linear regression fitting used {station_total} unique station locations statewide and their {date} recorded and/or estimated {minmax} temperatures (degC). Please consult each county map meta-data files for more details on map production and accuracy at the county scale. A leave one out cross validation (LOOCV) of the all station data used in all four counties produced individual R-squared values of: {rsqCounty} for Hawaii, Kauai, Maui (Maui, Lanai, Molokai, & Kahoolawe), and Honolulu (Oahu) counties respectively. As a whole leave one out cross validation (LOOCV) data from all stations data compared to observed monthly temperature (degC) produces a statewide R-squared value of: {rsqState} meaning overall this {date} statewide mosaic monthly {minmax} temperature map is a {quality} estimate of monthly temperature. All maps are subject to change as new data becomes available or unknown errors are corrected in reoccurring versions. Errors in {minmax} temperature estimates do vary over space meaning any gridded temperature value, even on higher quality maps, could still produce incorrect estimates. Check standard error (SE) maps to better understand spatial estimates of prediction error.'
    dataStatement_val = dataStatement_val.format(date=fmt_date,minmax=min_max[mode],station_total=str(station_total),
                                                rsqCounty=', '.join([str(np.round(x,2)) for x in meta['r2'][1:]]),rsqState=str(np.round(meta['r2'][0],2)),
                                                quality=quality)
    field_value_list = {'attribute':'value','dataStatement':dataStatement_val,'keywords':keyword_val,
                         'county':', '.join(icode_list),'dataYearMon':year+'-'+mon,'dateProduced':today,'dataVersionType':'archival',
                         'tempStationFile':temp_station_name,'tempGridFile':temp_tiff_name,'tempSEGridFile':temp_se_name,
                        'GeoCoordUnits':'Decimal Degrees','GeoCoordRefSystem':'+proj=longlat +datum=WGS84 +no_defs',
                        'XResolution':xres,'YResolution':yres,'ExtentXmin':xmin,'ExtentXmax':xmax,'ExtentYmin':ymin,
                        'ExtentYmax':ymax,'stationCountCounties':', '.join([str(x) for x in meta['StationCountCounties']]),'gridPixCounties':grid_pix,
                        'rsqTempCounties':' '.join([str(np.round(x,5)) for x in meta['r2'][1:]]),
                        'rmseTempCounties':' '.join([str(np.round(x,5)) for x in meta['rmse'][1:]]),
                        'maeTempCounties':' '.join([str(np.round(x,5)) for x in meta['mae'][1:]]),
                        'biasTempCounties':' '.join([str(np.round(x,5)) for x in meta['bias'][1:]]),
                        'stationCount':str(station_total),'gridPixCount':str(grid_pix_total),'rsqTemp':str(np.round(meta['r2'][0],5)),
                        'rmseTemp':str(np.round(meta['rmse'][0],5)),'maeTemp':str(np.round(meta['mae'][0],5)),
                        'biasTemp':str(np.round(meta['bias'][0],5)),'credits':credit_statement,'contacts':contact_list}
                    
    col1 = list(field_value_list.keys())
    col2 = [field_value_list[key] for key in col1]
    fmeta = open(meta_file,'w')
    for (key,val) in zip(col1,col2):
        line = [key,val]
        fmt_line = "{:20}{:60}\n".format(*line)
        fmeta.write(fmt_line)
    fmeta.close()
#END FUNCTIONS-----------------------------------------------------------------

if __name__ == '__main__':
    varname = sys.argv[1]
    run_master_dir = sys.argv[2]
    date_range = sys.argv[-1]

    st_date = pd.to_datetime(date_range.split('-')[0])
    en_date = pd.to_datetime(date_range.split('-')[1])
    date_list = pd.date_range(st_date,en_date)
    monthly_dates = [dt.to_timestamp() for dt in np.unique([dt.to_period('M') for dt in date_list])]

    mode = varname[1:]

    for dt in monthly_dates:
        date_str = dt.strftime('%Y-%m-%d')
        print(date_str)

        try:
            statewide_mosaic(varname,date_str)
            statewide_mosaic(varname,date_str,'_se')
            create_tables('T',mode,date_str)
            print('Done')
        except:
            print('Error')



