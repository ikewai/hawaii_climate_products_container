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
DEP_MASTER_DIR = MASTER_DIR + r'air_temp/monthly/dependencies/'
PRED_DIR = DEP_MASTER_DIR + r'predictors/'
MASK_TIFF_DIR = DEP_MASTER_DIR + r'geoTiffs_250m/masks/'
CV_OUTPUT_DIR = RUN_MASTER_DIR + r'tables/loocv/monthly/county/'
META_OUTPUT_DIR = RUN_MASTER_DIR + r'metadata/monthly/county/'
RAW_DATA_DIR = RUN_MASTER_DIR + r'tables/station_data/daily/raw_qc/statewide/'
MASTER_META_FILE = r'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'
TARG_STN_LIST = ['39.0','339.6','885.7','1075.0']
GAPFILL_DIR = DEP_MASTER_DIR + r'gapfill_models/'
GAPFILL_REF_YR = '20140101-20181231'
EXCL_LIST = {'Tmin':[],'Tmax':[]}
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

def get_isl_dims(iCode):
    tiffname = MASK_TIFF_DIR + iCode.lower() + '_mask.tif'
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
        cols = ['ObservedTemp','PredictedTemp','Obs-Pred','ValidatedStation']
        print(cv_data)
        sr = pd.Series([y_obs,y_pred,anom,'TRUE'],index=cols)
        cv_data.loc[target,cols] = sr

    #cv_data now populated for all target island stations
    #Include non-validated training data
    non_target_isl = prediction[~prediction['Island'].isin(isl_list)].index.values
    cv_data.loc[non_target_isl,'ObservedTemp'] = prediction.loc[non_target_isl,varname]
    cv_data.loc[non_target_isl,'ValidatedStation'] = 'FALSE'
    
    return cv_data

def get_metrics(varname,iCode,date_str,param_list=['dem_250'],inversion=2150):
    #Needs to makeModel based on the cv_data to get model parameters
    #Also needs to open some island reference file for dimensions
    """
    Requirements:
        --
    """
    if iCode == 'MN':
        isl_list = ['MA','MO','KO','LA']
    else:
        isl_list = [iCode]
    year = date_str.split('-')[0]
    mon = date_str.split('-')[1]
    date_tail = year+mon
    n_params = len(param_list)
    #File names
    temp_file = '_'.join((varname,'map',iCode,date_tail,'monthly')) + '.tif'
    se_file = '_'.join((varname,'map',iCode,date_tail,'se_monthly')) + '.tif'
    cv_file = CV_OUTPUT_DIR + iCode.upper() + '/' + '_'.join((date_tail,varname,iCode,'loocv_monthly')) + '.csv'
    pred_file = PRED_DIR + varname.lower() + '_predictors.csv'

    if varname == 'Tmean':
        tmin_file = '_'.join(('daily','Tmin',year,mon)) + '.csv'
        tmax_file = '_'.join(('daily','Tmax',year,mon)) + '.csv'
        input_file = ', '.join((tmin_file,tmax_file))
    else:
        input_file = '_'.join(('daily',varname,year,mon)) +'.csv'
    

    cv_data = pd.read_csv(cv_file)
    cv_data = cv_data.set_index('SKN')
    #Get actual linear regression info, don't make a dummy one for Tmean
    if varname == 'Tmean':
        theta = np.array([np.nan,np.nan,np.nan])
    else:
        pred_df,pr_series = tmpl.extract_predictors(pred_file,param_list)
        training_temp = cv_data['ObservedTemp']
        training_pred = pr_series.loc[training_temp.index]
        MODEL = tmpl.myModel(inversion=inversion)
        theta,pcov,X,y = tmpl.makeModel(training_temp,training_pred,MODEL,threshold=3)

    #Get island dims
    isl_dims = get_isl_dims(iCode)

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

def write_meta_text(mode,date_str,meta,version_type):
    year = date_str.split('-')[0]
    mon = date_str.split('-')[1]
    date_tail = year + mon
    varname = 'T'+mode
    formatted_date = pd.to_datetime(date_str).strftime('%b. %Y')
    isl_dict = {'BI':'Big Island','MA':'Maui','OA':'Oahu','KA':'Kauai','MN':'Maui, Molokai, Lanai, Kahoolawe'}
    island = meta['Island']
    cv_file = '_'.join((date_tail,varname,island.upper(),'loocv_monthly')) + '.csv'
    meta_file = META_OUTPUT_DIR + island +'/' + '_'.join((date_tail,varname,island.upper(),'meta_monthly')) + '.txt'

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
    dataStatement_val = 'This {date} monthly temperature {mode} map of {county} is a high spatial resolution gridded prediction of {mode} temperature in degrees Celsius for the month {date}. This was produced using a piece-wise linear regression model regressed on elevation with the junction point at {inversion} meters. ' + high_elev_statement + ' ' + r2_statement + ' All maps are subject to change as new data becomes available or unknown errors are corrected in reoccurring versions. Errors in temperature estimates do vary over space meaning any gridded temperature value, even on higher quality maps, could still produce incorrect estimates. Check standard error (SE) maps to better understand spatial estimates of prediction error'
    dataStatement_val = dataStatement_val.format(date=formatted_date,mode=mode,county=county_list,inversion=str(meta['inversion']))
    
    #Format keywords and credits
    kw_list = ', '.join([county_list,'Hawaii',mode+' temperature prediction','daily temperature','temperature','climate','linear regression'])
    
    credit_statement = 'All data produced by University of Hawaii at Manoa Dept. of Geography and the Enviroment, Ecohydology Lab in collaboration with the Water Resource Research Center (WRRC). Support for the Hawai‘i EPSCoR Program is provided by the Hawaii Emergency Management Agency.'
    contact_list = 'Keri Kodama (kodamak8@hawaii.edu), Matthew Lucas (mplucas@hawaii.edu), Ryan Longman (rlongman@hawaii.edu), Sayed Bateni (smbateni@hawaii.edu), Thomas Giambelluca (thomas@hawaii.edu)'
    
    #Arrange all meta fields and write to file
    field_value_list = {'attribute':'value','dataStatement':dataStatement_val,
                        'keywords':kw_list,'county':island.lower(),
                        'dataYearMon':formatted_date,
                        'dataVersionType':version_type,
                        'tempStationFile':meta['input_file'],
                        'tempGridFile':meta['temp_file'],
                        'tempSEGridFile':meta['se_file'],
                        'crossValidationFile':cv_file,
                        'GeoCoordUnits':'Decimal Degrees',
                        'GeoCoordRefSystem':'+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0',
                        'XResolution':str(meta['XResolution']),
                        'YResolution':str(meta['YResolution']),
                        'ExtentXmin':str(meta['Xmin']),
                        'ExtentXmax':str(meta['Xmax']),
                        'ExtentYmin':str(meta['Ymin']),
                        'ExtentYmax':str(meta['Ymax']),
                        'stationCount':str(meta['nstn']),
                        'outerStationCount':str(meta['nstn_ext']),
                        'regressionConst': str(np.round(regress_const,4)),
                        'regressionSlope1':str(np.round(regress_slope1,4)),
                        'regressionSlope2':str(np.round(regress_slope2,4)),
                        'biasTemp':str(np.round(meta['bias'],5)),
                        'rsqTemp':str(np.round(meta['R2'],5)),
                        'rmseTemp':str(np.round(meta['RMSE'],5)),
                        'maeTemp':str(np.round(meta['MAE'],5)),
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

def linear_gapfill(month_df,varname):
    #Check every station that needs to be filled
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
                pass
                #This is the climatological gapfill
            else:
                donor_stn = gapfill_stns[0]
                intercept = gapfill_df.at[donor_stn,'beta0']
                slope = gapfill_df.at[donor_stn,'beta1']
                donor_fill = tmpl.linear(month_df.at[donor_stn,varname],slope,intercept)
                filled_month.at[float(stn),varname] = donor_fill #Fills with estimated value
    
    return filled_month


def select_stations(month_df,varname,iCode,mix_high_alt=None,min_stn=10):
    """
    Fill
    """
    all_iCodes = ['BI','MA','KO','MO','LA','OA','KA']
    if iCode == 'MN':
        isl_list = ['MA','MO','KO','LA']
    else:
        isl_list = [iCode]
    
    nearest_neighbors = {'BI':['MA','OA','all'],'MN':['BI','OA','all'],'OA':['MA','KA','all'],'KA':['OA','MA','all']}
    
    filled_month = linear_gapfill(month_df,varname)

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

def main_monthly_cv(varname,date_str,iCode,params=['dem_250'],inversion=2150):
    year = date_str.split('-')[0]
    mon = date_str.split('-')[1]
    temp_file = RAW_DATA_DIR + '_'.join(('daily',varname,year,mon,'qc')) + '.csv'
    pred_file = PRED_DIR + varname.lower() + '_predictors.csv'
    cv_file = CV_OUTPUT_DIR + iCode.upper() + '/' + '_'.join((year+mon,varname,iCode,'loocv_monthly')) + '.csv'
    if iCode in ['BI','MN']:
        mix_high_alt = 2150
    else:
        mix_high_alt = None

    temp_df,temp_meta,temp_data = tmpl.extract_temp_input(temp_file)
    pred_df,pr_series = tmpl.extract_predictors(pred_file,params)
    month_temp = monthly_mean(temp_data,varname)
    month_isl = select_stations(month_temp,varname,iCode,mix_high_alt)
    pred_mon = pr_series.loc[month_isl.index]
    MODEL = tmpl.myModel(inversion=inversion)
    cv_data = cross_validation(month_isl,pred_mon,varname,MODEL,iCode,threshold=3)

    meta_master_df = pd.read_csv(MASTER_META_FILE)
    meta_master_df.set_index('SKN',inplace=True)
    cv_meta = meta_master_df.loc[cv_data.index]

    cv_full = cv_meta.join(cv_data,how='left')
    cv_full.reset_index(inplace=True)
    cv_full = cv_full.fillna('NA')
    cv_full.to_csv(cv_file,index=False)
    return cv_full

def main_tmean_monthly_cv(date_str,iCode):
    """
    Requirements: YYYYMM_Tmin_[iCode]_loocv_monthly.csv and 
    YYYYMM_Tmax_[iCode]_loocv_monthly.csv must exist
    """
    year = date_str.split('-')[0]
    mon = date_str.split('-')[1]
    tmin_loocv_file = CV_OUTPUT_DIR + iCode.upper() + '/' + '_'.join((year+mon,'Tmin',iCode,'loocv_monthly')) + '.csv'
    tmax_loocv_file = CV_OUTPUT_DIR + iCode.upper() + '/' + '_'.join((year+mon,'Tmax',iCode,'loocv_monthly')) + '.csv'
    tmean_loocv_file = CV_OUTPUT_DIR + iCode.upper() + '/' + '_'.join((year+mon,'Tmean',iCode,'loocv_monthly')) + '.csv'

    meta_table = pd.read_csv(MASTER_META_FILE)
    meta_table = meta_table.set_index('SKN')
    cv_tmin = pd.read_csv(tmin_loocv_file)
    cv_tmin = cv_tmin.set_index('SKN')
    cv_tmax = pd.read_csv(tmax_loocv_file)
    cv_tmax = cv_tmax.set_index('SKN')

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

    meta_tmean = meta_table.loc[shared_inds]
    cv_tmean = meta_tmean.join(cv_tmean,how='left')
    
    cv_tmean = cv_tmean.reset_index()
    cv_tmean.to_csv(tmean_loocv_file,index=False)

    return cv_tmean

#END FUNCTIONS-----------------------------------------------------------------

if __name__ == '__main__':
    varname = sys.argv[1]
    iCode = sys.argv[2]
    version_type = sys.argv[3]
    date_range = sys.argv[4]
    st_date = date_range.split('-')[0]
    en_date = date_range.split('-')[1]
    date_list = pd.date_range(st_date,en_date)
    monthly_dates = [dt.to_timestamp() for dt in np.unique([dt.to_period('M') for dt in date_list])]

    iCode = iCode.upper()
    mode = varname[1:]
    for dt in monthly_dates:
        date_str = dt.strftime('%Y-%m-%d')
        print(date_str)
        
        try:
            if varname == 'Tmean':
                cv_data = main_tmean_monthly_cv(date_str,iCode)
            else:
                cv_data = main_monthly_cv(varname,date_str,iCode)

            meta = get_metrics(varname,iCode,date_str)
            meta_file = write_meta_text(mode,date_str,meta,version_type)
            print('Done')
        except:
            print('Error')
