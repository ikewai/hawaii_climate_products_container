#Updated 3/31/22 version 2.1
#--Added climatological gapfill implementation
#--Adjusted directories for closing-gap (will need to update again for nrt)
import sys
import rasterio
import numpy as np
import pandas as pd

import Temp_linear as tmpl
from osgeo import gdal
from affine import Affine
from pyproj import Transformer


#DEFINE CONSTANTS--------------------------------------------------------------
MASTER_DIR = r'/home/hawaii_climate_products_container/preliminary/air_temp/'
RUN_MASTER_DIR = MASTER_DIR + r'data_outputs/'
DEP_MASTER_DIR = MASTER_DIR + r'monthly/dependencies/'
PRED_DIR = DEP_MASTER_DIR + r'predictors/'
PARAM_TIFF_DIR = DEP_MASTER_DIR + r'geoTiffs_250m/' #Fixed dir for location of parameter geotiffs
RAW_DATA_DIR = RUN_MASTER_DIR + r'tables/station_data/daily/raw_qc/statewide/'
MAP_OUTPUT_DIR = RUN_MASTER_DIR + r'tiffs/monthly/county/'
MASTER_META_FILE = r'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'
TARG_STN_LIST = ['39.0','339.6','885.7','1075.0']
GAPFILL_DIR = DEP_MASTER_DIR + r'gapfill_models/'
CLIM_DIR = DEP_MASTER_DIR + r'clim/'
GAPFILL_REF_YR = '20140101-20181231'
EXCL_LIST = {'Tmin':[],'Tmax':[]}
NO_DATA_VAL = -9999
#END CONSTANTS-----------------------------------------------------------------

#DEFINE FUNCTIONS--------------------------------------------------------------
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


# In[4]:


def genParamCode(param):

    if param == 'coastDist':
        return 'coastDistM'
    elif param == 'rf':
        return 'meanAnn'
    else:
        return param


# In[5]:


def genTiffName(param='dem', iCode='bi'):

    iCode = iCode.lower()
    if param == 'dem_250':
        param = 'dem'

    pCODE = genParamCode(param)
    TiffName = PARAM_TIFF_DIR + param + \
        '/' + iCode + '_' + pCODE + '_250m.tif'

    return TiffName


# In[6]:


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

def get_island_df(tiff_name,varname):
    lon,lat = get_Coordinates(tiff_name)
    lon = lon.reshape(-1)
    lat = lat.reshape(-1)

    df_dict = {'LON':lon,'LAT':lat}
    raster_img = rasterio.open(tiff_name)
    raster_data = raster_img.read(1)
    raster_mask = raster_img.read_masks(1)

    raster_mask[raster_mask > 0] = 1
    masked_array = raster_data * raster_mask

    masked_array[raster_mask == 0] = np.nan

    masked_array = masked_array.reshape(-1)

    df_dict[varname] = masked_array
    island_df = pd.DataFrame.from_dict(df_dict)
    shape = raster_data.shape

    return island_df, shape


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


# In[8]:

def G_islandName(iCode):

    if iCode == 'bi':
        return "Big Island"
    elif iCode == 'oa':
        return "Oahu"
    elif iCode == 'mn':
        return "Maui+"
    elif iCode == 'ka':
        return "Kauai"
    else:
        return iCode

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
    outdata.GetRasterBand(1).SetNoDataValue(NO_DATA_VAL)
    outdata.FlushCache()  # saves to disk!!
    outdata = None
    band = None
    ds = None

def linear_gapfill(month_df,varname,month):
    #Check every station that needs to be filled
    clim_name = CLIM_DIR + varname + '_stn_clim.csv'
    filled_month = month_df.copy() #Edit this df but reference month_df for checks
    #Ensure that gapfilled values aren't being used to estimate another gap value
    for stn in TARG_STN_LIST:
        if np.isnan(month_df.at[float(stn),varname]):
            gapfill_file = GAPFILL_DIR + varname + '_targetSKN'+ stn + '_' + GAPFILL_REF_YR + '.csv'
            gapfill_df = pd.read_csv(gapfill_file,skiprows=3,index_col=0)
            gapfill_df = gapfill_df.set_index('SKN')
            fill_stns = gapfill_df[gapfill_df['Significant']==True].index.values
            mon_nonan = month_df[varname].dropna().index.values
            gapfill_stns = fill_stns[np.isin(fill_stns,mon_nonan)]
            if len(gapfill_stns) == 0:
                #Climo gapfill, same as daily since it's a monthly climo file
                #Condition: none of the donor stations have available data
                clim_df = pd.read_csv(clim_name)
                mon = month - 1
                filled_month.at[float(stn),varname] = clim_df.at[mon,stn]
            else:
                donor_stn = gapfill_stns[0]
                intercept = gapfill_df.at[donor_stn,'beta0']
                slope = gapfill_df.at[donor_stn,'beta1']
                donor_fill = tmpl.linear(month_df.at[donor_stn,varname],slope,intercept)
                filled_month.at[float(stn),varname] = donor_fill #Fills with estimated value
    
    return filled_month


def select_stations(month_df,varname,iCode,month,mix_high_alt=None,min_stn=10):
    """
    Fill
    """
    clim_name = CLIM_DIR + varname + '_stn_clim.csv'
    master_df = pd.read_csv(MASTER_META_FILE)
    master_df = master_df.set_index('SKN')

    all_iCodes = ['BI','MA','KO','MO','LA','OA','KA']
    if iCode == 'MN':
        isl_list = ['MA','MO','KO','LA']
    else:
        isl_list = [iCode]
    
    nearest_neighbors = {'BI':['MA','OA','all'],'MN':['BI','OA','all'],'OA':['MA','KA','all'],'KA':['OA','MA','all']}
    
    filled_month = linear_gapfill(month_df,varname,month)

    if mix_high_alt == None:
        month_isl = filled_month[filled_month['Island'].isin(isl_list)]
    else:
        month_isl = filled_month[(filled_month['Island'].isin(isl_list)) | (filled_month['ELEV.m.'] > mix_high_alt)]
    
    #Check how many stations on target island
    month_isl = month_isl[~month_isl[varname].isna()]
    month_isl = month_isl.loc[~month_isl.index.isin(EXCL_LIST[varname])]
    hierarchy = nearest_neighbors[iCode]
    while ((month_isl.shape[0] < min_stn) & (set(isl_list) != set(all_iCodes))):
        next_isl = [hierarchy.pop(0)]
        if next_isl == ['all']:
            next_isl = list(set(all_iCodes) - set(isl_list))

        isl_list = isl_list + next_isl
        month_isl = pd.concat([month_isl,filled_month[filled_month['Island'].isin(next_isl)]])
        month_isl = month_isl[~month_isl[varname].isna()]
        #Exclude any additional stations in the exclusion list
        month_isl = month_isl.loc[~month_isl.index.isin(EXCL_LIST[varname])]

    #Final fill if still not enough stations (climatological emergency fill)
    if month_isl.shape[0] < min_stn:
        clim_df = pd.read_csv(clim_name)
        mon_ind = month - 1
        clim_df_skns = clim_df.columns.values.astype(float)
        non_overlap_skns = np.setdiff1d(clim_df_skns,month_isl.index.values)
        non_overlap_cols = [str(skn) for skn in non_overlap_skns]
        clim_to_join = clim_df.loc[mon_ind,non_overlap_cols]
        #Convert into consistent data frame/series
        clim_to_join.name = varname
        clim_to_join.index.name = 'SKN'
        clim_inds = clim_to_join.index.values.astype(float)
        clim_to_join = pd.DataFrame(clim_to_join).reset_index()
        clim_to_join['SKN'] = clim_to_join['SKN'].values.astype(float)
        clim_meta = master_df.loc[clim_inds]
        month_isl = month_isl.reset_index()
        #Merge ensures values go to their correct skns
        month_isl = month_isl.merge(clim_to_join,on=['SKN',varname],how='outer')
        month_isl = month_isl.set_index('SKN')
        month_isl.loc[clim_inds,clim_meta.columns] = clim_meta
        month_isl = month_isl.sort_index()
        
    return month_isl
    

def monthly_mean(temp_data,varname,threshold=15):
    #Get monthly mean of temperature for specified island
    valid_counts = temp_data.count(axis=1)
    stns_threshold = valid_counts[valid_counts>=threshold].index.values
    valid_mean = temp_data.loc[stns_threshold].mean(axis=1).rename(varname)

    temp_meta_master = pd.read_csv(MASTER_META_FILE)
    temp_meta_master.set_index('SKN',inplace=True)
    temp_inds = temp_meta_master.index.values
    month_df = pd.DataFrame(index=temp_inds,columns=[varname])
    month_df.loc[stns_threshold,varname] = valid_mean
    month_df = temp_meta_master.join(month_df,how='left')
    month_df[varname] = pd.to_numeric(month_df[varname])
    return month_df

def minmax_main(varname,iCode,date_str,params=['dem_250'],inversion=2150):
    """
    Full monthly mean temperature grid for 1 month of data
    """
    threshold = 2.5
    station_dir = RAW_DATA_DIR
    pred_dir = PRED_DIR
    output_dir = MAP_OUTPUT_DIR + varname + '/' + iCode.upper() + '/'
    output_se_dir = MAP_OUTPUT_DIR + varname + '_se/' + iCode.upper() + '/'
    year = date_str.split('-')[0]
    mon = date_str.split('-')[1]
    temp_file = station_dir + '_'.join(('daily',varname,year,mon,'qc')) + '.csv'
    pred_file = pred_dir + varname.lower() + '_predictors.csv'
    Tmap_tiff = varname + '_map_' + iCode + '_' + year + mon + '_monthly.tif'
    Tmap_tiff_path = output_dir + Tmap_tiff
    se_tiff = varname + '_map_' + iCode + '_' + year + mon + '_monthly_se.tif'
    se_tiff_path = output_se_dir + se_tiff
    if iCode in ['BI','MN']:
        mix_high_alt = 2150
    else:
        mix_high_alt = None
    
    temp_df,temp_meta,temp_data = tmpl.extract_temp_input(temp_file)
    pred_df,pr_series = tmpl.extract_predictors(pred_file,params)
    MODEL = tmpl.myModel(inversion=inversion)
    month_temp = monthly_mean(temp_data,varname)
    month_isl = select_stations(month_temp,varname,iCode,int(mon),mix_high_alt)
    pred_mon = pr_series.loc[month_isl.index]
    theta,pcov,X,y = tmpl.makeModel(month_isl[varname],pred_mon,MODEL,threshold=threshold)
    island_df,mask,shape = get_island_grid(iCode,params)
    se_model = tmpl.get_std_error(pred_mon,month_isl[varname],pcov,island_df[params],inversion)

    X_island = island_df.values
    T_model = MODEL(X_island[:, 2:], *theta)
    T_model[mask == 0] = NO_DATA_VAL
    se_model[mask == 0] = NO_DATA_VAL

    #Output the temp map
    input_tiff_name = genTiffName(iCode=iCode)
    output_tiff(T_model,input_tiff_name,Tmap_tiff_path,shape)

    #Output SE map
    output_tiff(se_model,input_tiff_name,se_tiff_path,shape)
    #Note for self, not including meta data here as it is contained in another wrapper now

    return (T_model,se_model,theta,pcov)

def mean_main(iCode,date_str):
    min_varname = 'Tmin'
    max_varname = 'Tmax'
    year = date_str.split('-')[0]
    mon = date_str.split('-')[1]
    date_tail = year + mon
    tmax_dir = MAP_OUTPUT_DIR + max_varname + '/' + iCode.upper() + '/'
    tmin_dir = MAP_OUTPUT_DIR + min_varname + '/' + iCode.upper() + '/'
    tmax_se_dir = MAP_OUTPUT_DIR + max_varname + '_se/' + iCode.upper() + '/'
    tmin_se_dir = MAP_OUTPUT_DIR + min_varname + '_se/' + iCode.upper() + '/'
    tmean_dir = MAP_OUTPUT_DIR + 'Tmean/' + iCode.upper() + '/'
    tmean_se_dir = MAP_OUTPUT_DIR + 'Tmean_se/' + iCode.upper() + '/'

    #Gridded temperature
    tmin_tiff_name = tmin_dir + '_'.join((min_varname,'map',iCode,date_tail,'monthly')) + '.tif'
    tmax_tiff_name = tmax_dir + '_'.join((max_varname,'map',iCode,date_tail,'monthly')) + '.tif'

    tmean_tiff_name = tmean_dir + '_'.join(('Tmean','map',iCode,date_tail,'monthly')) + '.tif'

    min_df, shape = get_island_df(tmin_tiff_name,min_varname)
    max_df, shape = get_island_df(tmax_tiff_name,max_varname)

    merged_df = min_df.merge(max_df,how='inner',on=['LON','LAT'])

    tmean = merged_df[[min_varname,max_varname]].mean(axis=1).values
    tmean[np.isnan(tmean)] = NO_DATA_VAL

    #Output to new tiff
    output_tiff(tmean,tmin_tiff_name,tmean_tiff_name,shape)

    #Gridded standard error
    tmin_se_name = tmin_se_dir + '_'.join((min_varname,'map',iCode,date_tail,'monthly','se')) + '.tif'
    tmax_se_name = tmax_se_dir + '_'.join((max_varname,'map',iCode,date_tail,'monthly','se')) + '.tif'
    tmean_se_name = tmean_se_dir + '_'.join(('Tmean','map',iCode,date_tail,'monthly','se')) + '.tif'

    se_min_df, shape = get_island_df(tmin_se_name,min_varname)
    se_max_df, shape = get_island_df(tmax_se_name,max_varname)

    se_min2 = se_min_df[min_varname]**2
    se_max2 = se_max_df[max_varname]**2
    se_mean = np.sqrt((se_min2) + (se_max2)) / 2.0

    se_mean[np.isnan(se_mean)] = NO_DATA_VAL

    output_tiff(se_mean.values,tmin_se_name,tmean_se_name,shape)

    return tmean,se_mean


    
#END FUNCTIONS-----------------------------------------------------------------

if __name__=='__main__':
    varname = sys.argv[1]
    iCode = sys.argv[2]
    date_range = sys.argv[3]

    st_date = pd.to_datetime(date_range.split('-')[0])
    en_date = pd.to_datetime(date_range.split('-')[1])

    date_list = pd.date_range(st_date,en_date)
    #Convert it to monthly series
    monthly_dates = [dt.to_timestamp() for dt in np.unique([dt.to_period('M') for dt in date_list])]

    for mdate in monthly_dates:
        date_str = mdate.strftime('%Y-%m-%d')
        print(iCode,date_str)
        try:
            if varname == 'Tmean':
                Tmean, smean = mean_main(iCode.upper(),date_str)
            else:
                T_model,se_model,theta,pcov = minmax_main(varname,iCode.upper(),date_str)
            print('Done')
        except:
            print('Error')
