#Daily only
import sys
import rasterio
import numpy as np
import pandas as pd

import Temp_linear as tmpl
from affine import Affine
from pyproj import Transformer

#DEFINE CONSTANTS--------------------------------------------------------------
MASTER_DIR = r'/home/hawaii_climate_products_container/preliminary/'
RUN_MASTER_DIR = MASTER_DIR + r'air_temp/data_outputs/'
DEP_MASTER_DIR = MASTER_DIR + r'air_temp/daily/dependencies/'
PRED_DIR = DEP_MASTER_DIR + r'predictors/'
MASK_TIFF_DIR = DEP_MASTER_DIR + r'geoTiffs_250m/masks/'
CV_OUTPUT_DIR = RUN_MASTER_DIR + r'tables/loocv/daily/county/'
META_OUTPUT_DIR = RUN_MASTER_DIR + r'metadata/daily/county/'
META_MASTER_FILE = r'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'
#END CONSTANTS-----------------------------------------------------------------

#DEFINE FUNCTIONS--------------------------------------------------------------
def get_coordinates(GeoTiff_name):

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

def get_isl_dims(iCode,mask_tiff_dir):
    """
    Helper function
    Dependencies: input_dir = parent input directory branch
    """
    tiffname = mask_tiff_dir + iCode.lower() + '_mask.tif'
    lons,lats = get_coordinates(tiffname)
    lons = np.unique(lons.reshape(-1))
    lats = np.unique(lats.reshape(-1))
    xdiff = lons[1:]-lons[:-1]
    ydiff = lats[1:]-lats[:-1]
    xresolution = np.round(np.min(xdiff),6)
    yresolution = np.round(np.min(ydiff),6)
    xmin = np.min(lons)
    xmax = np.max(lons)
    ymin = np.min(lats)
    ymax = np.max(lats)
    isl_dims = {'XResolution':xresolution,'YResolution':yresolution,'Xmin':xmin,'Xmax':xmax,'Ymin':ymin,'Ymax':ymax}

    return isl_dims

def cross_validation(prediction,predictor,varname,model,iCode,threshold=2.5):
    """
    Input requirements:
    prediction: dataframe including, at minimum, Island and [varname] columns, indexed by SKN
    predictor: series including model predictor columns, indexed by SKN
    """
    #Get target_isl stations (only validate against target island stations, not supplementary stations)
    if iCode == 'MN':
        isl_list = ['MA','MO','LA','KO']
    else:
        isl_list = [iCode]
    
    target_isl = prediction[prediction['Island'].isin(isl_list)].index.values
    
    cv_data = pd.DataFrame(index=prediction.index)
    for target in target_isl:
        #All stations excluding target station
        train_inds = np.setdiff1d(predictor.index.values,[target])
        X_train = predictor.loc[train_inds]
        X_test = predictor.loc[target].values.reshape(-1,1)
        y_train = prediction.loc[train_inds,varname]
        y_obs = prediction.at[target,varname]
        theta,pcov,X,y = tmpl.makeModel(y_train,X_train,model,threshold)
        y_pred = model(X_test,*theta)
        anom = y_obs - y_pred
        cv_data.loc[target,['ObservedTemp','PredictedTemp','Obs-Pred','ValidatedStation']] = [y_obs,y_pred,anom,'TRUE']

    #cv_data now populated for all target island stations
    #Include non-validated training data
    non_target_isl = prediction[~prediction['Island'].isin(isl_list)].index.values
    cv_data.loc[non_target_isl,'ObservedTemp'] = prediction.loc[non_target_isl,varname]
    cv_data.loc[non_target_isl,'ValidatedStation'] = 'FALSE'
    
    return cv_data

def get_metrics(varname,iCode,date_str,param_list,inversion=2150):
    #Needs to makeModel based on the cv_data to get model parameters
    #Also needs to open some island reference file for dimensions
    """
    Requirements:
        --
    """
    cv_dir = CV_OUTPUT_DIR + iCode.upper() + '/'
    pred_dir = PRED_DIR
    mask_dir = MASK_TIFF_DIR
    if iCode == 'MN':
        isl_list = ['MA','MO','KO','LA']
    else:
        isl_list = [iCode]
    date_tail = ''.join(date_str.split('-'))
    year = date_str.split('-')[0]
    mon = date_str.split('-')[1]
    n_params = len(param_list)
    #File names
    temp_file = '_'.join((varname,'map',iCode,date_tail)) + '.tif'
    se_file = '_'.join((varname,'map',iCode,date_tail,'se')) + '.tif'
    cv_file = cv_dir + '_'.join((date_tail,varname,iCode,'loocv')) + '.csv'
    pred_file = pred_dir + varname.lower() + '_predictors.csv'

    if varname == 'Tmean':
        tmin_file = '_'.join(('daily','Tmin',year,mon)) + '.csv'
        tmax_file = '_'.join(('daily','Tmax',year,mon)) + '.csv'
        input_file = ', '.join((tmin_file,tmax_file))
    input_file = '_'.join(('daily',varname,year,mon)) +'.csv'
    

    cv_data = pd.read_csv(cv_file)
    cv_data.set_index('SKN',inplace=True)
    #Get actual linear regression info, don't make a dummy one for Tmean
    if varname == 'Tmean':
        theta = np.array([np.nan,np.nan,np.nan])
    else:
        pred_df,pr_series = tmpl.extract_predictors(pred_file,param_list)
        training_temp = cv_data['ObservedTemp']
        training_pred = pr_series.loc[training_temp.index]
        MODEL = tmpl.myModel(inversion=inversion)
        theta,pcov,X,y = tmpl.makeModel(training_temp,training_pred,MODEL,threshold=2.5)

    #Get island dims
    isl_dims = get_isl_dims(iCode,mask_dir)

    #Number of stations
    non_target_stns = cv_data[~cv_data['Island'].isin(isl_list)]
    non_target_isl_codes = cv_data[~cv_data['Island'].isin(isl_list)]['Island'].unique()
    high_elev_stns = non_target_stns[non_target_stns['ELEV.m.'] > inversion]
    high_elev_isl_codes = high_elev_stns['Island'].unique()
    nstn = cv_data.shape[0]
    nstn_ext = non_target_stns.shape[0]
    nstn_elev = high_elev_stns.shape[0]

    #Check the numbers
    observed = cv_data[cv_data['ValidatedStation']==True]['ObservedTemp'].values.flatten()
    predicted = cv_data[cv_data['ValidatedStation']==True]['PredictedTemp'].values.flatten()
    pred_clip,obs_clip = tmpl.sigma_Clip(predicted,observed)
    if ((len(pred_clip) - n_params -1) < 3) | ((len(obs_clip) - n_params - 1) < 3):
        obs_mean = np.nan
        pred_mean = np.nan
        mae = np.nan
        rmse = np.nan
        r2 = np.nan
        aic = np.nan
        aicc = np.nan
        bic = np.nan
        bias = np.nan
        r2_code = 1 #Not enough data to produce R2
    else:
        mae,rmse,r2,aic,aicc,bic = tmpl.metrics(pred_clip,obs_clip,False,n_params)
        obs_mean = np.mean(observed)
        pred_mean = np.mean(predicted)
        bias = obs_mean - pred_mean
        if r2 >= 0:
            r2_code = 0
        else:
            r2_code = 2 #negative R2

    meta = {'Island':iCode,'inversion':inversion,'nstn':nstn,'nstn_ext':nstn_ext,'nstn_elev':nstn_elev,'outer_islands':non_target_isl_codes,'high_islands':high_elev_isl_codes,
            'obs_mean':obs_mean,'pred_mean':pred_mean,'bias':bias,'MAE':mae,'RMSE':rmse,'R2':r2,'AIC':aic,'AICc':aicc,'BIC':bic,'r2_code':r2_code,'input_file':input_file,'temp_file':temp_file,'se_file':se_file,'lr_coef':theta}
    
    meta = {**meta,**isl_dims}
    return meta

def write_meta_text(varname,date_str,meta):
    date_tail = ''.join((date_str.split('-')))
    formatted_date = pd.to_datetime(date_str).strftime('%b. %d, %Y')
    temp_mode = {'Tmin':'minimum','Tmax':'maximum','Tmean':'mean'}
    isl_dict = {'BI':'Big Island','MA':'Maui','OA':'Oahu','KA':'Kauai','MN':'Maui, Molokai, Lanai, Kahoolawe'}
    island = meta['Island']
    meta_dir = META_OUTPUT_DIR + island.upper() + '/'
    cv_file = '_'.join((date_tail,varname,island.upper(),'loocv')) + '.csv'
    meta_file = meta_dir + '_'.join((date_tail,varname,island.upper(),'meta')) + '.txt'


    if island == 'BI':
        county_list = 'Hawaii County'
    elif island == 'MN':
        county_list = 'Maui County (Maui, Lanai, Molokai, Kahoolawe)'
    elif island == 'OA':
        county_list = 'Honolulu County (Oahu)'
    elif island == 'KA':
        county_list = 'Kauai County'
    
    if meta['nstn_elev'] > 0:
        high_isl_list = list(meta['high_islands'])
        high_islands = [isl_dict[icode.upper()] for icode in high_isl_list]
        high_islands = ', '.join(high_islands)
    if meta['nstn_ext'] > 0:
        outer_isl_list = list(meta['outer_islands'])
        outer_islands = [isl_dict[icode.upper()] for icode in outer_isl_list]
        outer_islands = ', '.join(outer_islands)

    #Mixed station text case
    if (meta['nstn_ext'] > meta['nstn_elev']) & (meta['nstn_elev'] > 0):
        high_elev_statement = 'The model was trained on {nstn} unique station location(s) within {county} and supplemented at high elevation by {nstn_elev} station(s) from {high_islands}. Due to limited station availability, the model training was also supplemented by {nstn_ext} station(s) drawn from {outer_islands}.'
        high_elev_statement = high_elev_statement.format(nstn=str(meta['nstn']),county=county_list,nstn_elev=str(meta['nstn_elev']),high_islands=high_islands,nstn_ext=str(meta['nstn_ext']),outer_islands=outer_islands)
    elif meta['nstn_ext'] > 0:
        high_elev_statement = 'The model was trained on {nstn} unique station location(s) within {county}. Due to limited station availability, the model training was supplemented by {nstn_ext} station(s) from {outer_islands}.'
        high_elev_statement = high_elev_statement.format(nstn=str(meta['nstn']),county=county_list,nstn_ext=str(meta['nstn_ext']),outer_islands=outer_islands)
    elif (meta['nstn_ext'] == meta['nstn_elev']) & (meta['nstn_ext'] > 0):
        high_elev_statement = 'The model was trained on {nstn} unique station location(s) within {county} and supplemented at high elevation by {nstn_elev} station(s) from {high_islands}.'
        high_elev_statement = high_elev_statement.format(nstn=str(meta['nstn']),county=county_list,nstn_elev=str(meta['nstn_elev']),high_islands=high_islands)
    else:
        high_elev_statement = 'The model was trained on {nstn} unique station location(s) within {county}.'
        high_elev_statement = high_elev_statement.format(nstn=meta['nstn'],county=county_list)
    
    lr_coef = meta['lr_coef']
    regress_const = lr_coef[0]
    regress_slope1 = lr_coef[1]
    if len(lr_coef) > 2:
        regress_slope2 = lr_coef[2]
    else:
        regress_slope2 = np.nan
    
    if meta['r2_code'] == 1:
        r2_statement = 'Insufficient validation stations were available for the target island. Leave-one-out cross-validation (LOOCV) could not be performed and R-squared value is nan.'
    elif meta['r2_code'] == 2:
        r2_statement = 'A leave-one-out cross-validation (LOOCV) was performed based on the station data available for the target island. However, the R-squared value is negative. If outer island data were used to supplement the model training, R-squared may not accurately represent goodness of fit. Please consult the cross-validation table or the standard error maps for more information on model error.'
    else:
        r2_statement = 'A leave one out cross validation (LOOCV) of the station data used in this map produced an R-squared of: {rsqr}.'
        r2_statement = r2_statement.format(rsqr=str(np.round(meta['R2'],4)))

    #Format data statement
    dataStatement_val = 'This {date} daily temperature {mode} map of {county} is a high spatial resolution gridded prediction of {mode} temperature in degrees Celsius for the date {date}. This was produced using a piece-wise linear regression model regressed on elevation with the junction point at {inversion} meters. ' + high_elev_statement + ' ' + r2_statement + ' All maps are subject to change as new data becomes available or unknown errors are corrected in reoccurring versions. Errors in temperature estimates do vary over space meaning any gridded temperature value, even on higher quality maps, could still produce incorrect estimates. Check standard error (SE) maps to better understand spatial estimates of prediction error'
    dataStatement_val = dataStatement_val.format(date=formatted_date,mode=temp_mode[varname],county=county_list,inversion=str(meta['inversion']))
    
    #Format keywords and credits
    kw_list = ', '.join([county_list,'Hawaii',temp_mode[varname]+' temperature prediction','daily temperature','temperature','climate','linear regression'])
    
    credit_statement = 'All data produced by University of Hawaii at Manoa Dept. of Geography and the Enviroment, Ecohydology Lab in collaboration with the Water Resource Research Center (WRRC). Support for the Hawai‘i EPSCoR Program is provided by the Hawaii Emergency Management Agency.'
    contact_list = 'Keri Kodama (kodamak8@hawaii.edu), Matthew Lucas (mplucas@hawaii.edu), Ryan Longman (rlongman@hawaii.edu), Sayed Bateni (smbateni@hawaii.edu), Thomas Giambelluca (thomas@hawaii.edu)'
    
    #Arrange all meta fields and write to file
    field_value_list = {'attribute':'value','dataStatement':dataStatement_val,'keywords':kw_list,
        'county':island.lower(),'dataDate':formatted_date,'dataVersionType':'preliminary','tempStationFile':meta['input_file'],'tempGridFile':meta['temp_file'],
        'tempSEGridFile':meta['se_file'],'crossValidationFile':cv_file,'fillValue':'-9999','GeoCoordUnits':'Decimal Degrees',
        'GeoCoordRefSystem':'+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0','XResolution':str(meta['XResolution']),
        'YResolution':str(meta['YResolution']),'ExtentXmin':str(meta['Xmin']),
        'ExtentXmax':str(meta['Xmax']),'ExtentYmin':str(meta['Ymin']),
        'ExtentYmax':str(meta['Ymax']),'stationCount':str(meta['nstn']),
        'outerStationCount':str(meta['nstn_ext']),'regressionConst': str(np.round(regress_const,4)),'regressionSlope1':str(np.round(regress_slope1,4)),'regressionSlope2':str(np.round(regress_slope2,4)),'biasTemp':str(np.round(meta['bias'],5)),'rsqTemp':str(np.round(meta['R2'],5)),
        'rmseTemp':str(np.round(meta['RMSE'],5)),'maeTemp':str(np.round(meta['MAE'],5)),
        'credits':credit_statement,'contacts':contact_list}
    col1 = list(field_value_list.keys())
    col2 = [field_value_list[key] for key in col1]
    fmeta = open(meta_file,'w')
    for (key,val) in zip(col1,col2):
        line = [key,val]
        fmt_line = "{:20}{:60}\n".format(*line)
        fmeta.write(fmt_line)
    fmeta.close()
    return meta_file

def main_cv_single(varname,date_str,temp_data,pred_data,iCode,output_dir=None,inversion=2150):
    """
    Outputs cross-validation table for single specified county and variable.
    Cannot accommodate Tmean which requires both variables
    """
    date_tail = ''.join(date_str.split('-'))
    loocv_name = output_dir + iCode.upper() + '/' + '_'.join((date_tail,varname,iCode,'loocv')) + '.csv'
    MODEL = tmpl.myModel(inversion)

    cv_temp = cross_validation(temp_data,pred_data,varname,MODEL,iCode)
    
    temp_meta = pd.read_csv(META_MASTER_FILE)
    temp_meta.set_index('SKN',inplace=True)
    cv_meta = temp_meta.loc[cv_temp.index]
    
    cv_temp = cv_meta.join(cv_temp,how='left')
    cv_temp.reset_index(inplace=True)
    cv_temp.to_csv(loocv_name,index=False)
    
    return cv_temp

def main_cv_mean(date_str,iCode,cv_dir=CV_OUTPUT_DIR):
    date_tail = ''.join(date_str.split('-'))
    varname = 'Tmean'

    #input files
    tmin_file = cv_dir + iCode.upper() + '/' + '_'.join((date_tail,'Tmin',iCode,'loocv')) + '.csv'
    tmax_file = cv_dir + iCode.upper() + '/' + '_'.join((date_tail,'Tmax',iCode,'loocv')) + '.csv'

    #Output file
    tmean_loocv_file = cv_dir + iCode.upper() + '/' + '_'.join((date_tail,varname,iCode,'loocv')) + '.csv'
    
    meta_master_table = pd.read_csv(META_MASTER_FILE)
    meta_master_table = meta_master_table.set_index('SKN')

    cv_tmin = pd.read_csv(tmin_file)
    cv_tmin = cv_tmin.set_index('SKN')
    cv_tmax = pd.read_csv(tmax_file)
    cv_tmax = cv_tmax.set_index('SKN')

    shared_inds = list(set(cv_tmin.index.values) & set(cv_tmax.index.values))
    obs_tmin = cv_tmin.loc[shared_inds,'ObservedTemp']
    obs_tmax = cv_tmax.loc[shared_inds,'ObservedTemp']
    obs_tmean = (obs_tmin + obs_tmax) * 0.5
    pred_tmin = cv_tmin.loc[shared_inds,'PredictedTemp']
    pred_tmax = cv_tmax.loc[shared_inds,'PredictedTemp']
    pred_tmean = (pred_tmin + pred_tmax) * 0.5

    cv_tmean = pd.DataFrame(index=shared_inds)
    cv_tmean.index.name = 'SKN'
    cv_tmean.loc[shared_inds,'ObservedTemp'] = obs_tmean
    cv_tmean.loc[shared_inds,'PredictedTemp'] = pred_tmean
    cv_tmean.loc[shared_inds,'Obs-Pred'] = obs_tmean - pred_tmean
    valid_inds = cv_tmean['PredictedTemp'].dropna().index.values
    cv_tmean.loc[cv_tmean.index.isin(valid_inds),'ValidatedStation'] = 'TRUE'
    cv_tmean.loc[~cv_tmean.index.isin(valid_inds),'ValidatedStation'] = 'FALSE'

    meta_tmean = meta_master_table.loc[shared_inds]
    cv_tmean = meta_tmean.join(cv_tmean,how='left')
    cv_tmean = cv_tmean.sort_values(by='SKN')
    
    cv_tmean = cv_tmean.reset_index()
    cv_tmean.to_csv(tmean_loocv_file,index=False)
    return cv_tmean


def main_cv_all(date_str,tmin_df,tmax_df,param_tmin,param_tmax,iCode,meta_table,output_dir,inversion=2150):
    """
    Outputs cross-validation tables for tmin, tmax, tmean and metadata dictionaries
    """
    date_tail = ''.join(date_str.split('-'))
    tmin_name = output_dir + iCode.upper() + '/' + '_'.join((date_tail,'Tmin',iCode,'loocv')) + '.csv'
    tmax_name = output_dir + iCode.upper() + '/' + '_'.join((date_tail,'Tmax',iCode,'loocv')) + '.csv'
    tmean_name = output_dir + iCode.upper() + '/' + '_'.join((date_tail,'Tmean',iCode,'loocv')) + '.csv'
    MODEL = tmpl.myModel(inversion)
    cv_tmin = cross_validation(tmin_df,param_tmin,'Tmin',MODEL,iCode)
    cv_tmax = cross_validation(tmax_df,param_tmax,'Tmax',MODEL,iCode)

    #For stations with both Tmin and Tmax available, compute predicted and observed Tmean
    #Predicted Tmean_i defined as mean(pred_tmin_i,pred_tmax_i)
    #Observed Tmean_i defined as mean(obs_tmin_i,obs_tmax_i)
    shared_inds = list(set(cv_tmin.index.values) & set(cv_tmax.index.values))
    obs_tmin = cv_tmin.loc[shared_inds,'ObservedTemp']
    obs_tmax = cv_tmax.loc[shared_inds,'ObservedTemp']
    obs_tmean = (obs_tmin + obs_tmax) * 0.5
    pred_tmin = cv_tmin.loc[shared_inds,'PredictedTemp']
    pred_tmax = cv_tmax.loc[shared_inds,'PredictedTemp']
    pred_tmean = (pred_tmin + pred_tmax) * 0.5
    cv_tmean = pd.DataFrame(index=shared_inds)
    cv_tmean.loc[shared_inds,'ObservedTemp'] = obs_tmean
    cv_tmean.loc[shared_inds,'PredictedTemp'] = pred_tmean
    cv_tmean.loc[shared_inds,'Obs-Pred'] = obs_tmean - pred_tmean
    valid_inds = cv_tmean['PredictedTemp'].dropna().index.values
    cv_tmean.loc[cv_tmean.index.isin(valid_inds),'ValidatedStation'] = 'TRUE'
    cv_tmean.loc[~cv_tmean.index.isin(valid_inds),'ValidatedStation'] = 'FALSE'

    meta_table = meta_table.set_index('SKN')
    tmin_meta = meta_table.loc[cv_tmin.index]
    tmax_meta = meta_table.loc[cv_tmax.index]
    tmean_meta = meta_table.loc[cv_tmean.index]

    cv_tmin = tmin_meta.join(cv_tmin,how='left')
    cv_tmax = tmax_meta.join(cv_tmax,how='left')
    cv_tmean = tmean_meta.join(cv_tmean,how='left')

    #Write the files
    #Tmin
    cv_tmin = cv_tmin.reset_index()
    cv_tmin.to_csv(tmin_name,index=False)

    #Tmax
    cv_tmax = cv_tmax.reset_index()
    cv_tmax.to_csv(tmax_name,index=False)

    #Tmean
    cv_tmean = cv_tmean.reset_index()
    cv_tmean.to_csv(tmean_name,index=False)

    return (cv_tmin,cv_tmax,cv_tmean)

    


    
#END FUNCTIONS-----------------------------------------------------------------

#MAIN

if __name__ == '__main__':
    varname = sys.argv[1]
    iCode = sys.argv[2]
    master_dir = sys.argv[3]
    run_version = sys.argv[4]
    version_type = sys.argv[5]
    date_range = sys.argv[6] #YYYYMMDD_st-YYYYMMDD_en
    date_range = date_range.split('-')
    st_date = pd.to_datetime(date_range[0])
    en_date = pd.to_datetime(date_range[-1])
    dt_range = pd.date_range(st_date,en_date)

    params = ['dem_250']
    run_master_dir = master_dir + 'finalRunOutputs' + run_version + '/' + version_type + '/'
    temp_input_dir = run_master_dir + 'tables/' + varname + '_daily_raw/'
    proc_input_dir = master_dir + 'input/'
    cv_dir = run_master_dir + 'tables/loocv/county/'
    pred_dir = proc_input_dir + 'predictors/'
    pred_file = pred_dir + varname.lower() + '_predictors.csv'
    
    for dt in dt_range:
        date_str = dt.strftime('%Y-%m-%d')
        print(date_str)
        year_str = date_str.split('-')[0]
        mon_str = date_str.split('-')[1]

        temp_file = temp_input_dir + '_'.join(('daily',varname,year_str,mon_str)) + '.csv'
        temp_df,temp_meta,temp_data = tmpl.extract_temp_input(temp_file)
        pred_df,pr_series = tmpl.extract_predictors(pred_file,params)

        temp_date = tmpl.get_temperature_date(temp_data,temp_meta,iCode,date_str,varname=varname)
        pred_temp = pr_series.loc[temp_date.index]

        cv_temp = main_cv_single(varname,date_str,temp_date,pred_temp,iCode,cv_dir)
        
        temp_meta = get_metrics(varname,iCode,date_str,proc_input_dir,run_master_dir,params)
        temp_text = write_meta_text(varname,date_str,temp_meta,run_master_dir)


