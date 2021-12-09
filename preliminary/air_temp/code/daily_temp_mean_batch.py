#Creates temperature mean from Tmin and Tmax average
import sys
import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
from affine import Affine
from pyproj import Transformer


#NAMING SETTINGS & OUTPUT FLAGS----------------------------------------------#
MASTER_DIR = r'/home/kodamak8/share/air_temp/'
CODE_MASTER_DIR = MASTER_DIR + r'code/'
WORKING_MASTER_DIR = MASTER_DIR + r'working_data/'
RUN_MASTER_DIR = MASTER_DIR + r'preliminary_output/'
MAP_OUTPUT_DIR = RUN_MASTER_DIR + r'tiffs/county/temp/' #Location of geotiff/png output
SE_OUTPUT_DIR = RUN_MASTER_DIR + r'tiffs/county/temp_SE/'
CV_OUTPUT_DIR = RUN_MASTER_DIR + r'loocv/county/'
TIFF_SUFFIX = '.tif'
SE_SUFFIX = '_se.tif'
CV_SUFFIX = '_loocv.csv'
#END SETTINGS----------------------------------------------------------------#

#FUNCTION DEFINITION---------------------------------------------------------#
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

    masked_array[raster_mask == 0] = 0

    masked_array = masked_array.reshape(-1)

    df_dict[varname] = masked_array
    island_df = pd.DataFrame.from_dict(df_dict)
    shape = raster_data.shape

    return island_df, shape

def output_tiff(data,base_tiff_name,out_tiff_name,tiff_shape):
    cols,rows = tiff_shape
    ds = gdal.Open(base_tiff_name)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(out_tiff_name, rows, cols, 1, gdal.GDT_Float64)
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

#END FUNCTIONS---------------------------------------------------------------#
#Open raster files for Tmin and Tmax
#Merge standard error as well

#"Main functions"
def generate_temp_mean(iCode,date_str,datadir):
    date_tail = ''.join(date_str.split('-'))
    iCode = iCode.upper()
    #Set tiff filename
    min_varname = 'Tmin'
    max_varname = 'Tmax'
    tmin_tiff_name = datadir + '_'.join((min_varname,'map',iCode,date_tail)) + TIFF_SUFFIX
    tmax_tiff_name = datadir + '_'.join((max_varname,'map',iCode,date_tail)) + TIFF_SUFFIX

    tmean_tiff_name = datadir + '_'.join(('Tmean','map',iCode,date_tail)) + TIFF_SUFFIX

    #Open raster files and convert to dataframes

    min_df, shape = get_island_df(tmin_tiff_name,min_varname)
    max_df, shape = get_island_df(tmax_tiff_name,max_varname)

    #Merge them so that we have Tmin and Tmax in same data frame only at matching lat-lons
    #(Although lat-lons should not have any mismatch)
    merged_df = min_df.merge(max_df,how='inner',on=['LON','LAT'])

    #Take the mean of Tmin and Tmax columns, convert into dataframe
    tmean = merged_df[[min_varname,max_varname]].mean(axis=1).values

    #Output to new tiff
    output_tiff(tmean,tmin_tiff_name,tmean_tiff_name,shape)
    
def generate_se_mean(iCode,date_str,datadir,metadir):

    #Setting file names
    date_tail = ''.join(date_str.split('-'))
    iCode = iCode.upper()
    min_varname = 'Tmin'
    max_varname = 'Tmax'
    se_min_tiff_name = datadir + '_'.join((min_varname,'map',iCode,date_tail)) + SE_SUFFIX
    se_max_tiff_name = datadir + '_'.join((max_varname,'map',iCode,date_tail)) + SE_SUFFIX
    
    se_mean_tiff_name = datadir + '_'.join(('Tmean','map',iCode,date_tail)) + SE_SUFFIX
    #Reading tiff data and getting statistic values
    se_min_df, tiff_shape = get_island_df(se_min_tiff_name,min_varname)
    se_max_df, tiff_shape = get_island_df(se_max_tiff_name,max_varname)


    #Create an array from the combined standard deviations
    #Square each and divide by sample size (? what is the sample size?)
    se_min2 = se_min_df[min_varname]**2
    se_max2 = se_max_df[max_varname]**2
    combined = np.sqrt((se_min2) + (se_max2)) / 2.0

    output_tiff(combined.values,se_min_tiff_name,se_mean_tiff_name,tiff_shape)
    

# This code has been automatically covnerted to comply with the pep8 convention
# This the Linux command:
# $ autopep8 --in-place --aggressive  <filename>.py
if __name__ == '__main__':

    iCode = str(sys.argv[1])  # 'bi'
    #main_dir = sys.argv[2] #Parent dir, assuming standard temp file tree
    date_range = str(sys.argv[2]) #YYYYMMDD_start-YYYYMMDD_end

    date_range = date_range.split('-')
    st_date = pd.to_datetime(date_range[0],format='%Y%m%d')
    en_date = pd.to_datetime(date_range[1],format='%Y%m%d')

    date_list = pd.date_range(st_date,en_date)

    temp_dir = MAP_OUTPUT_DIR
    se_dir = SE_OUTPUT_DIR
    metadir = RUN_MASTER_DIR + 'tables/loocv/county/'

    iCode = iCode.upper()
    for dt in date_list:
        date_str = str(dt.date())
        print(iCode,'Tmean',date_str)
        try:
            generate_temp_mean(iCode,date_str,temp_dir)
            generate_se_mean(iCode,date_str,se_dir,metadir)
            print('Success------------^')
        except:
            print('Error -------------^ ')






