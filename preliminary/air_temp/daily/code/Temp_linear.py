#Version 2.0

#Version 1.1.3
#--Updated from development version: 6/24/21
#Description:
#Module toolkit used for the gridded temperature map production and post-processing
#Development notes:
#2021-06-24
#--Updated version to 1.1
#--Deprecated 1.0 versions of removeOutlier, get_predictors, and makeModel
#--Added new functions: get_temperature_date, select_stations, extract_data
#2021-07-02
#Updated to version 1.1.1
#--Fixed high elevation gap-fill indexing bug in get_temperature_date and select_stations
#--Set default mixHighAlt to 2150 instead of None (i.e. always include mixed island high elev stations)
#2021-07-09
#--Updated version to 1.1.2:
#--Added new function: get_std_error
#2021-07-12
#--Updated version to 1.1.3:
#--Added new function: lr_temp_gapfill
#--Adjusted select_stations. Restored mixHighAlt default to None. Value determined based on island.
#--Clim gapfill incorporated as last fallback for lr_temp_gapfill
#--Hardcoded constants declared at start of module. Edit as needed.
#2021-08-11
#--Minor patch: Corrected divide by zero case in cross-validation function. metrics(...) cannot run when validation station too low with respect to n_params
#--Tmax gapfill stations added

#from attr import field
#import pylab as py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


# In[ ]:
#SET MODULE CONSTANTS
#Consolidate file names, index names, and directory names here to avoid hardcoding
STN_IDX_NAME = 'SKN'
ELEV_IDX_NAME = 'ELEV.m.'
MASTER_DIR = r'/home/hawaii_climate_products_container/preliminary/'
DEP_MASTER_DIR = MASTER_DIR + r'air_temp/daily/dependencies/'
GP_DATA_DIR = DEP_MASTER_DIR + r'gapfill_models/'
CLIM_DATA_DIR = DEP_MASTER_DIR + r'clim/'
META_MASTER_FILE = r'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'
TARG_STN_LIST = ['39.0','339.6','885.7','1075.0']
#GAPFILL_PREF = ''
#GAPFILL_INF = ''
GAPFILL_SUF = '_20140101-20181231.csv'
#CLIM_PREF = ''
#CLIM_INF = ''
CLIM_SUF = '_stn_clim.csv'
PRED_STNS_MIN = {'39.0':['39.2','98.2','869.0','828.0','43.3','728.2','499.9','1036.0','94.0','107.4'],
    '339.6':['39.2','113.2','946.0','43.3','499.12','1036.0','499.9','885.7','499.13','63.0'],
    '885.7':['869.0','799.0','499.12','679','828.0','1074.0','1069.0','842.9','841.2','538.0'],
    '1075.0':['1069.0','1036.0','1074.0','538.0','858.0','842.8','828.0','752.6','842.9','742.4']}
PRED_STNS_MAX = {'39.0':['39.2','107.4','1069.0','855.3','311.2','48.0','95.6','1029.0','499.13','946.0'],
    '339.6':['39.2','107.4','267.8','499.13','129.0','75.1','266.0','147.2','752.6','1029.0'],
    '885.7':['869.0','828.0','499.12','909.0','742.4','752.6','858.0','541.2','911.1','842.7'],
    '1075.0':['499.6','3.9','266.0','43.3','63.0','499.8','1036.0','147.2','499.9','101.1']}
TMIN_STN_EXCLUDE = [728.2] #[728.2] Put this back
TMAX_STN_EXCLUDE = [728.2]

#END MODULE CONSTANTS--------------------------------------------------------------

def get_clim_file(varname):
    #Change according to file naming convention needs
    clim_name = CLIM_DATA_DIR + varname + CLIM_SUF
    return clim_name

def linear(x, a, b):
    return a * x + b


def bilinear(x, a, b, c):

    left = a * x + b
    right = c * (x - 2150) + (a * 2150 + b)

    try:
        y = np.asarray([left[i] if x[i] <= 2150 else right[i]
                        for i in range(len(x))])
        return y
    except BaseException:
        if x <= 2150:
            return left
        else:
            return right


# calculate bic for regression
def calculate_bic(n, mse, num_params):
    BIC = n * np.log(mse) + num_params * np.log(n)
    return BIC

# calculate aic for regression


def calculate_aic(n, mse, num_params):

    # for the linear regression, assuming that errors are normally distributed
    AIC = n * np.log(mse) + 2 * num_params

    AICc = AIC + 2 * num_params * (num_params + 1.) / (n - num_params - 1.)

    return AIC, AICc


def lr_temp_gapfill(isl_df,varname,stn_date):
    """
    Description: Helper function specific to linear gap-fill of temperature min/max
    Patch notes:
        --[10/5/21] Function breaks for new input where nan-row stations are dropped from file
        --          First checks index list to see if all gapfilled stations exist in index
        --          If not present, they are gapfilled automatically
    Development notes:
        --Iteratively checks all target stations (specified in module constants)
        --If target missing data, chooses predictor model based on highest correlation (specified)
        --If no predictor stations available, fill with climatological value at target station
    """
    if varname == 'Tmin':
        predictor_stations = PRED_STNS_MIN
    elif varname == 'Tmax':
        predictor_stations = PRED_STNS_MAX
    
    #Get list of all critical stations for gapfilling
    #Ensures an index exists for donor and target stations for gapfill check
    master_meta = pd.read_csv(META_MASTER_FILE)
    master_meta = master_meta.set_index('SKN')
    critical_stns = TARG_STN_LIST + [item for sublist in [predictor_stations[key] for key in predictor_stations.keys()] for item in sublist]
    critical_stns = [float(crit) for crit in critical_stns]
    non_exist_crits = np.setdiff1d(np.array(critical_stns),isl_df.index.values)
    non_exist_meta = master_meta.loc[non_exist_crits]
    new_inds = list(non_exist_crits) + list(isl_df.index.values)
    new_isl_df = pd.DataFrame(index=new_inds)
    new_isl_df.index.name = 'SKN'
    new_isl_df.loc[isl_df.index,isl_df.columns] = isl_df
    new_isl_df.loc[non_exist_crits,varname] = np.nan
    new_isl_df.loc[non_exist_crits,non_exist_meta.columns] = non_exist_meta
    #Check if target stations for gapfilling are part of the input dataset
    #Then run gapfill as normal
    for target in TARG_STN_LIST:
        if np.isnan(new_isl_df.at[float(target),varname]):
            #iteratively check the regression parameters
            fill_file = GP_DATA_DIR + varname + '_target' + STN_IDX_NAME + target + GAPFILL_SUF
            fill_model_df = pd.read_csv(fill_file, skiprows=3)
            fill_model_df = fill_model_df.set_index(STN_IDX_NAME)
            pred_stn_list = predictor_stations[target]
            for pred in pred_stn_list:
                #check if avail, if yes, predict and fill
                #if not, pass to next
                lr_fill_flag = False
                if np.isnan(new_isl_df.at[float(pred),varname]):
                    #Station not available. Move on.
                    pass
                else:
                    beta0 = fill_model_df.at[float(pred),'beta0']
                    beta1 = fill_model_df.at[float(pred),'beta1']
                    pred_x = new_isl_df.at[float(pred),varname]
                    targ_est = linear(pred_x,beta1,beta0)
                    isl_df.at[float(target),varname] = targ_est
                    lr_fill_flag = True
                    break
    #if no linear regression was used, fill target with climo
            if not lr_fill_flag:
                clim_file = get_clim_file(varname)
                clim_df = pd.read_csv(clim_file)
                mon = stn_date.month - 1
                new_isl_df.at[float(target),varname] = clim_df.at[mon,target]

    return new_isl_df
# In[ ]:
def removeOutlier(X,y,threshold=2.5):
    X = X.flatten()
    fit, cov = curve_fit(bilinear, X, y, sigma=y * 0 + 1)
    model = bilinear(X, fit[0], fit[1], fit[2])
    stdev = np.std(model - y)  # 1-sigma scatter of residuals
    indx, = np.where(np.abs(model - y) < threshold * stdev)

    # repeating the process one more time to clip outliers based
    # on a more robust model
    fit, cov = curve_fit(
        bilinear, X[indx], y[indx], sigma=y[indx] * 0 + 1)
    model = bilinear(X, fit[0], fit[1], fit[2])
    stdev = np.std(model - y)
    indx, = np.where(np.abs(model - y) < threshold * stdev)

    return indx

# In[ ]:
def select_stations(vars,varname,iCode,stn_date,min_stn=10,mixHighAlt=None):
    """
    Description: Primarily internal function to progressively sample stations from outer islands //
    as needed to meet minimum regression sample size
    Development notes:
        --Currently specifies distinct selection hierarchy for each island
        --Pulls high elevation stations from all islands as long as inversion height is specified
        --Replaces highest elevation station with climatological value if no high elevation data available
    Patch 2021-07-02:
        --Fixed indexing bug for high elevation climatological gap-fill
    Update 2021-07-12:
        --Introduced linear regression gap-filling
    Future patches:
        
    """
    #Input is already filtered by date. Single day station dataset, all islands
    #Sets decision algorithm for handling corner cases. May need to consider wrapping this
    #Filter temps based on iCode, check length, re-filter or return
    #Set exclusions
    if varname == 'Tmin':
        excl_list = TMIN_STN_EXCLUDE
    elif varname == 'Tmax':
        excl_list = TMAX_STN_EXCLUDE

    #Defining search hierarchy for each island (Add more or change order here as desired)
    all_iCodes = ['BI','MA','KO','MO','LA','OA','KA']
    ka_hier = ['OA','MA','All']
    oa_hier = ['KA','MA','All']
    ma_hier = ['BI','OA','All']
    bi_hier = ['MA','OA','All']
    mn_hier = ['MA','BI','OA','All']

    #Set original baseline island list
    if (iCode == 'MN'):
        isl_list = ['MA','MO','KO','LA']
        hierarchy = ma_hier
    elif iCode == 'BI':
        isl_list = [iCode]
        hierarchy = bi_hier
    elif iCode == 'MA':
        isl_list = [iCode]
        hierarchy = ma_hier
    elif iCode == 'OA':
        isl_list = [iCode]
        hierarchy = oa_hier
    elif iCode == 'KA':
        isl_list = [iCode]
        hierarchy = ka_hier
    elif iCode in ['MO','KO','LA']:
        isl_list = [iCode]
        hierarchy = mn_hier
    else:
        return None

    master_df = pd.read_csv(META_MASTER_FILE)
    master_df = master_df.set_index('SKN')
    #As long as inversion height is set by mixHighAlt parameter, automatically include all available
    #Automatically gapfill all pre-selected target stations
    var_isl = lr_temp_gapfill(vars,varname,stn_date)
    
    if mixHighAlt is not None:
        var_isl = var_isl[(var_isl['Island'].isin(isl_list) | (var_isl[ELEV_IDX_NAME] > mixHighAlt))]
    else:
        var_isl = var_isl[var_isl['Island'].isin(isl_list)]
        
    #Iteratively check number of available stations. Progressively add outer island stations until minimum requirement is met
    var_isl = var_isl[~var_isl[varname].isna()]
    #Exclude any stations in exclusion list
    var_isl = var_isl.loc[~var_isl.index.isin(excl_list)]
    while ((var_isl.shape[0] < min_stn) & (set(isl_list) != set(all_iCodes))):
        next_isl = [hierarchy.pop(0)]
        if next_isl == ['All']:
            next_isl = list(set(all_iCodes) - set(isl_list))

        isl_list = isl_list + next_isl
        var_isl = pd.concat([var_isl,vars[vars['Island'].isin(next_isl)]])
        var_isl = var_isl[~var_isl[varname].isna()]
        #Exclude any additional stations in the exclusion list
        var_isl = var_isl.loc[~var_isl.index.isin(excl_list)]
    
    #Final check if var_isl still below min_stn threshold.
    if var_isl.shape[0] < min_stn:
        clim_df = pd.read_csv(get_clim_file(varname))
        mon_ind = stn_date.month - 1
        clim_df_skns = clim_df.columns.values.astype(float)
        #only replace values that aren't already in var_isl
        non_overlap_skns = np.setdiff1d(clim_df_skns,var_isl.index.values)
        non_overlap_cols = [str(skn) for skn in non_overlap_skns]
        clim_to_join = clim_df.loc[mon_ind,non_overlap_cols]
        clim_to_join.name = varname
        clim_to_join.index.name = 'SKN'
        clim_inds = clim_to_join.index.values.astype(float)
        clim_to_join = pd.DataFrame(clim_to_join).reset_index()
        clim_to_join['SKN'] = clim_to_join['SKN'].values.astype(float)
        clim_meta = master_df.loc[clim_inds]
        var_isl = var_isl.reset_index()
        var_isl = var_isl.merge(clim_to_join,on=['SKN',varname],how='outer')
        var_isl = var_isl.set_index('SKN')
        var_isl.loc[clim_inds,clim_meta.columns] = clim_meta
        var_isl = var_isl.sort_index()
        
    var_isl = var_isl[~var_isl.index.duplicated(keep='first')]
    return var_isl

def extract_dataset(varname,dataloc='',predictors=True,pred_name=None,predloc=None):
    """
    Description: Simple dataset extraction. No data processing performed.
    Development notes:
        --Currently allows retrieval of data and related predictors. Will later need to generalize this functionality.
        --Really, should only output one specified dataset at a time and keep data processing in other specified functions.
    Future patches:
        --Remove hardcoded file suffixes or at least create more dynamic options
        --Either allow for other file types or specify this function is for csv extraction
    """
    #Extracts full dataset based on specified varname
    #Option to continuously add new variable handling
    if varname == 'Tmax':
        var_file = dataloc+varname+'_QC.csv'
    elif varname == 'Tmin':
        var_file = dataloc+varname+'_QC.csv'
    elif varname =='RF':
        var_file = dataloc+'2_Partial_Fill_Daily_RF_mm_1990_2020.csv'

    var_df = pd.read_csv(var_file, encoding="ISO-8859-1", engine='python')

    if predictors == True:
        if predloc is None:
            predloc = dataloc
        if pred_name is None:
            pred_name = varname
        pred_file = predloc+pred_name+'_predictors.csv'
        pred_df = pd.read_csv(pred_file, encoding="ISO-8859-1",engine='python')
        return var_df, pred_df
    else:
        return var_df 

# Need a process_archival function to convert non-standardized format data
def extract_predictors(filename,param_list):
    pred_df = pd.read_csv(filename,encoding="ISO-8859-1",engine='python')
    pred_df = pred_df.set_index(STN_IDX_NAME)
    return (pred_df,pred_df[param_list])

def extract_temp_input(filename,meta_col_n=12,get_decomp=True):
    """
    Reads the temperature input data for a specified date
    Processes it according to the date standard, outputs a meta-only dataframe (SKN-sorted),
    and a temp-only dataframe (SKN-sorted)
    """
    master_df = pd.read_csv(META_MASTER_FILE)
    master_df = master_df.set_index(STN_IDX_NAME)
    temp_df = pd.read_csv(filename,encoding="ISO-8859-1",engine='python')
    temp_df = temp_df.set_index(STN_IDX_NAME)
    df_cols = list(temp_df.columns)
    meta_cols = list(master_df.columns)
    temp_cols = [col for col in df_cols if col not in meta_cols]
    meta_df = temp_df[list(meta_cols)]
    temp_data = temp_df[list(temp_cols)]

    #Convert keys into datetime keys for easier time indexing
    temp_cols = [dt.split('X')[1] for dt in list(temp_cols)]
    dt_keys = pd.to_datetime(list(temp_cols))
    temp_data.columns = dt_keys

    temp_df = meta_df.join(temp_data,how='left')
    if get_decomp:
        return (temp_df,meta_df,temp_data)
    else:
        return temp_df

def get_temperature_date(temp_data,meta_data,iCode,stn_date,varname=None,climloc='',dateFmt=None,mixHighAlt=None,min_stn=10,naive_select=False):
    #Updated to take in a station-indexed temperature dataframe, should already be set_index(SKN)
    iCode = iCode.upper()
    if isinstance(stn_date,str):
        if dateFmt == None:
            stn_date = pd.to_datetime(stn_date)
        else:
            stn_date = pd.to_datetime(stn_date,format=dateFmt)
    
    temp_day = temp_data[[stn_date]].rename(columns={stn_date:varname})
    temp_day = meta_data.join(temp_day,how='left')
    #Send islands and temp_day into select_stations.
    #Outputs temp stn data of appropriate size
    #if mixHighAlt not specified, set mixHighAlt based on the target island
    #if mixHighAlt is specified, then force it to be the user specified value
    if mixHighAlt == None:
        if iCode in ['KA','OA']:
            mixHighAlt = None
        else:
            mixHighAlt = 2150
    if naive_select:
        #Only select all island data from specified date
        return temp_day.dropna()
    else:
        return select_stations(temp_day,varname,iCode,stn_date,min_stn=min_stn,mixHighAlt=mixHighAlt)

def get_predictors(pred_df,param_list):
    """
    Description: Updated version of get_Predictors
    Development notes:
        --Removed redundancy in ISLAND_code call
        --Now only outputs predictors which will actually be used in curve fitting
    Future patches:
        --Eventually will need to consider where to handle predictors from multiple sources
    """
    pred_df = pred_df.set_index(STN_IDX_NAME)
    return pred_df[param_list]


# In[ ]:

def myModel(inversion=2150):
    '''
    This wrapper function constructs another function called "MODEL"
    according to the provided inversion elevation
    '''

    def MODEL(X, *theta):

        _, n_params = X.shape

        y = theta[0] + theta[1] * X[:, 0]
        for t in range(1, n_params):
            y += theta[t+2] * X[:, t]

        ind, = np.where(X[:, 0] > inversion)
        y[ind] += theta[2] * (X[:, 0][ind] - inversion)

        return y

    return MODEL


# In[ ]:
def makeModel(predictand,params,model,threshold=2.5):
    """
    Description: Updated version of makeModel.
    Development notes:
        --Predictand replaces df for clarity. Only available and relevant stations of general predictand should be passed from this variable
        --Params replaces parameter list. Parameter list should be filtered for selected predictors before being passed in
        --Data preparation encapsulated in different function. This function now exclusively takes input and fits curve.
    """
    n_data, n_params = params.shape
    y = predictand.values
    X = params.values
    if len(y) > 1:
        indx = removeOutlier(X,y,threshold=threshold)
        X = X[indx]
        y = y[indx]
        fit, cov = curve_fit(model,X,y,p0=[30, -0.002] + (n_params) * [0])
        return fit, cov, X, y
    else:
        return None, None, None, None



def get_std_error(X,y,pcov,param_grid,inversion):
    """
    Description: Based on estimated parameter variance-covariance matrix
    computes the standard error of the model predicted values.
    Patch notes: Version 1.0
    """
    se_fit = []
    X_island = param_grid.copy()
    if np.isinf(pcov).any():
        #Remove outliers linear----------------------------------------
        threshold = 2.5
        Xvals = X.values.flatten()
        yvals = y.values
        fit, cov = curve_fit(linear, Xvals, yvals, sigma=yvals * 0 + 1)
        model = linear(Xvals, fit[0], fit[1])
        stdev = np.std(model - yvals)  # 1-sigma scatter of residuals
        indx, = np.where(np.abs(model - yvals) < threshold * stdev)

        fit, cov = curve_fit(
            linear, Xvals[indx], yvals[indx], sigma=yvals[indx] * 0 + 1)
        model = linear(Xvals, fit[0], fit[1])
        stdev = np.std(model - yvals)
        indx, = np.where(np.abs(model - yvals) < threshold * stdev)
        #Remove outliers end-------------------------------------------
        #indx = removeOutlier(X.values,y.values,threshold=2.5)
        X = X.iloc[indx]
        y = y.iloc[indx]
        se_model = sm.OLS(y,sm.add_constant(X))
        se_res = se_model.fit()
        pcov = se_res.cov_params().values
        X_island = sm.add_constant(X_island.values)
        for i in range(X_island.shape[0]):
            xi = X_island[i].reshape(-1,1)
            se = np.dot(np.dot(xi.T,pcov),xi)[0][0]
            se_fit.append(se)
        se_fit = np.array(se_fit)
    else:
        #X_island = sm.add_constant(param_grid.values)
        X_above = X_island['dem_250'].copy() - inversion
        X_above[X_above<=0] = 0
        X_above.rename('above')
        X_island.insert(1,'above',X_above)
        X_island = sm.add_constant(X_island.values)
        for i in range(X_island.shape[0]):
            xi = X_island[i].reshape(-1,1)
            se = np.dot(np.dot(xi.T,pcov),xi)[0][0]
            se_fit.append(se)
        
        se_fit = np.array(se_fit)
    
    return se_fit
# In[ ]:
def cross_validation(predictor, response, iCode, varname, MODEL, metadata, threshold=2.5,inversion=2150):
    if iCode == 'MN':
        isl_list = ['MA','KO','MO','LA']
    else:
        isl_list = [iCode]
    #Only select test values from target island
    meta_stn = metadata.set_index('SKN')
    targ_skns = []
    predicted_y = []
    validate_y = []
    target_isl = response[response['Island'].isin(isl_list)].index.values
    non_target_stn = response[~response['Island'].isin(isl_list)]
    non_target_isl = response[~response['Island'].isin(isl_list)]['Island'].unique()
    high_elev_stn = non_target_stn[non_target_stn['ELEV.m.'] > inversion]
    high_elev_isl = high_elev_stn['Island'].unique()
    nstn = response.shape[0]
    nstn_ext = len(non_target_stn)
    nstn_elev = len(high_elev_stn)

    for target in list(target_isl):
        train_inds = np.setdiff1d(predictor.index.values,[target])
        X_train = predictor.loc[train_inds]
        X_test = predictor.loc[target].values.reshape(-1,1)
        y_train = response.loc[train_inds,varname]
        y_test = response.loc[target,varname]
        theta,pcov,X,y = makeModel(y_train,X_train,MODEL,threshold)
        y_loo = MODEL(X_test,*theta)
        targ_skns.append(target)
        predicted_y.append(y_loo)
        validate_y.append(y_test)
    
    targ_skns = np.array(targ_skns).reshape(-1,1)
    predicted_y = np.array(predicted_y).reshape(-1,1)
    validate_y = np.array(validate_y).reshape(-1,1)
    validate_flag = np.ones(validate_y.shape,dtype=bool)
    anoms = validate_y - predicted_y
    cv_data = np.concatenate([targ_skns,validate_y,predicted_y,anoms,validate_flag],axis=1)
    n_params = X_train.shape[1]
    u,v = sigma_Clip(predicted_y.flatten(),validate_y.flatten())
    if ((len(u) - n_params -1) < 3) | ((len(v) - n_params - 1) < 3):
        mae = np.nan
        rmse = np.nan
        r2 = np.nan
        aic = np.nan
        aicc = np.nan
        bic = np.nan
        obs_mean = np.nan
        pred_mean = np.nan
        bias = np.nan
        r2_code = 1 #Not enough data to produce metric
    else:
        mae,rmse,r2,aic,aicc,bic = metrics(u,v,False,n_params)
        obs_mean = np.mean(v)
        pred_mean = np.mean(u)
        bias = obs_mean - pred_mean
        if r2 >= 0:
            r2_code = 0
        else:
            r2_code = 2 #Negative R2
    
    #Convert the arrays to dataframe (add the other columns as we figure out what they are)
    cv_df = pd.DataFrame(cv_data,columns=[STN_IDX_NAME,'ObservedTemp','PredictedTemp','Obs-Pred','ValidatedStation'])
    cv_meta = meta_stn.loc[cv_df[STN_IDX_NAME].values]
    cv_meta = cv_meta.reset_index()

    cv_df = pd.concat([cv_df[STN_IDX_NAME],cv_meta,cv_df[cv_df.columns[1:]]],axis=1)
    cv_df = cv_df.loc[:,~cv_df.columns.duplicated()]
    #Tack on the values for the training-only values from off-island if applicable
    train_only_inds = np.setdiff1d(predictor.index.values,target_isl)
    train_meta = meta_stn.loc[train_only_inds]
    train_meta = train_meta.reset_index()
    
    train_only_validate = response.loc[train_only_inds,varname].values
    train_only_predicted = np.array([np.nan for i in range(train_only_validate.shape[0])])
    training_flag = np.zeros(train_only_predicted.shape,dtype=bool)
    
    train_only_data = np.concatenate([train_only_inds.reshape(-1,1),train_only_validate.reshape(-1,1),train_only_predicted.reshape(-1,1),train_only_predicted.reshape(-1,1),training_flag.reshape(-1,1)],axis=1)

    train_only_df = pd.DataFrame(train_only_data,columns=['SKN','ObservedTemp','PredictedTemp','Obs-Pred','ValidatedStation'])
    train_only_df = pd.concat([train_only_df[STN_IDX_NAME],train_meta,train_only_df[train_only_df.columns[1:]]],axis=1)
    train_only_df = train_only_df.loc[:,~train_only_df.columns.duplicated()]

    cv_df = pd.concat([cv_df,train_only_df],axis=0)
    booleanDictionary = {True: 'TRUE', False: 'FALSE'}
    cv_df['ValidatedStation'] = cv_df['ValidatedStation'].map(booleanDictionary)
    #cv_df = cv_df.set_index(STN_IDX_NAME)
    meta = {'Island':iCode,'inversion':inversion,'nstn':nstn,'nstn_ext':nstn_ext,'nstn_elev':nstn_elev,'outer_islands':non_target_isl,'high_islands':high_elev_isl,'obs_mean':obs_mean,'pred_mean':pred_mean,'bias':bias,'MAE':mae,'RMSE':rmse,'R2':r2,'AIC':aic,'AICc':aicc,'BIC':bic,'r2_code':r2_code}

    return cv_df, meta

# calcualte metrics based on a leave one out strategy

def metrics(y1, y2, verbose=False, n_param=1, n_data=None):
    '''
    y1 and y2 are two series of the same size

    This function outputs the MAE, RMSE and R^2
    of the cross evaluated series.

    '''
    y1 = y1.reshape(-1)
    y2 = y2.reshape(-1)

    if n_data is None:
        n_data = len(y1)

    mse = np.mean((y1 - y2)**2)

    RMSE = np.sqrt(mse)
    MAE = np.mean(np.abs(y1 - y2))
    R2 = np.max([r2_score(y1, y2), r2_score(y2, y1)])

    BIC = calculate_bic(n_data, mse, n_param)
    AIC, AICc = calculate_aic(n_data, mse, n_param)

    if verbose:
        print('MAE: %.2f' % MAE, ' RMSE: %.2f' % RMSE, ' R^2: %.2f' % R2)
        print('AIC: %.2f' % AIC, 'AIC: %.2f' % AICc, ' BIC: %.2f' % BIC)

    return MAE, RMSE, R2, AIC, AICc, BIC
# In[ ]:

def sigma_Clip(u, v, threshold=3.0):

    # removing 10% upper and lower quantiles of residuals (removing aggressive
    # outliers)
    delta = u - v
    indx = np.argsort(delta)
    u = u[indx]
    v = v[indx]
    N = len(u)
    i = int(np.ceil(1 * N / 10))
    j = int(np.floor(9 * N / 10))
    u = u[i:j]
    v = v[i:j]

    # Here we do some sigma clipping (assuming that residuals are normally
    # distributed)
    delta = u - v
    mean = np.median(delta)
    std = np.std(delta)
    indx = (
        (delta > mean -
         threshold *
         std) & (
            delta < mean +
            threshold *
            std))
    u = u[indx]
    v = v[indx]

    return u, v

def write_meta_data(meta,date_str,mode,temp_file,se_file,cv_file,meta_file='meta.txt',lr_coef=None,isl_dims=None,mix_station_case=0,inversion=2150):
    date = pd.to_datetime(date_str).strftime('%b. %-d, %Y')
    island = meta['Island']
    stn_file = 'T'+mode + '_QC.csv'

    isl_dict = {'BI': 'Big Island','MA':'Maui','OA':'Oahu','KA':'Kauai'}
    #Set island name text
    if island == 'BI':
        county_list = 'Hawaii County'
    elif island == 'MN':
        county_list = 'Maui County (Maui, Lanai, Molokai, Kahoolawe)'
    elif island == 'OA':
        county_list = 'Honolulu County (Oahu)'
    elif island == 'KA':
        county_list = 'Kauai County'
    
    #Get outer island values if applicable
    if meta['nstn_elev'] > 0:
        high_isl_list = list(meta['high_islands'])
        high_islands = [isl_dict[icode] for icode in high_isl_list]
        high_islands = ', '.join(high_islands)
    if meta['nstn_ext'] > 0:
        outer_isl_list = list(meta['outer_islands'])
        outer_islands = [isl_dict[icode] for icode in outer_isl_list]
        outer_islands = ', '.join(outer_islands)
 
    #Set statement for station locations and their count
    if mix_station_case == 0:
        #No mixed stations
        high_elev_statement = 'The model was trained on {nstn} unique station location(s) within {island}.'
        high_elev_statement = high_elev_statement.format(nstn=meta['nstn'],island=county_list)
    elif mix_station_case == 1:
        #High elevation mixed only
        high_elev_statement = 'The model was trained on {nstn} unique station location(s) within {island} and supplemented at high elevation by {nstn_elev} station(s) from {high_islands}.'
        high_elev_statement = high_elev_statement.format(nstn=str(meta['nstn']),island=county_list,nstn_elev=str(meta['nstn_elev']),high_islands=high_islands)
    elif mix_station_case == 2:
        #Outer island mixed only
        high_elev_statement = 'The model was trained on {nstn} unique station location(s) within {island}. Due to limited station availability, the model training was supplemented by {nstn_ext} station(s) from {outer_islands}.'
        high_elev_statement = high_elev_statement.format(nstn=str(meta['nstn']),island=county_list,nstn_ext=str(meta['nstn_ext']),outer_islands=outer_islands)
    elif mix_station_case == 3:
        #Outer island mixed with high elevation mix
        high_elev_statement = 'The model was trained on {nstn} unique station location(s) within {island} and supplemented at high elevation by {nstn_elev} station(s) from {high_islands}. Due to limited station availability, the model training was also supplemented by {nstn_ext} station(s) drawn from {outer_islands}.'
        high_elev_statement = high_elev_statement.format(nstn=str(meta['nstn']),island=county_list,nstn_elev=str(meta['nstn_elev']),high_islands=high_islands,nstn_ext=str(meta['nstn_ext']),outer_islands=outer_islands)
    
    #Set fields for regression coefficient
    regress_const = lr_coef[0]
    regress_slope1 = lr_coef[1]
    if len(lr_coef) > 2:
        regress_slope2 = lr_coef[2]
    else:
        regress_slope2 = np.nan
    
    #Set statement for Rsquared value
    if meta['r2_code'] == 1:
        r2_statement = 'Insufficient validation stations were available for the target island. Leave-one-out cross-validation (LOOCV) could not be performed and R-squared value is nan.'
    elif meta['r2_code'] == 2:
        r2_statement = 'A leave-one-out cross-validation (LOOCV) was performed based on the station data available for the target island. However, the R-squared value is negative. If outer island data were used to supplement the model training, R-squared may not accurately represent goodness of fit. Please consult the cross-validation table or the standard error maps for more information on model error.'
    else:
        r2_statement = 'A leave one out cross validation (LOOCV) of the station data used in this map produced an R-squared of: {rsqr}.'
        r2_statement = r2_statement.format(rsqr=str(np.round(meta['R2'],4)))

    #Set general data statement fields
    dataStatement_val = 'This {date} daily temperature {mode} map of {island} is a high spatial resolution gridded prediction of {mode} temperature in degrees Celsius for the date {date}. This was produced using a piece-wise linear regression model regressed on elevation with the junction point at {inversion} meters. ' + high_elev_statement + ' ' + r2_statement + ' All maps are subject to change as new data becomes available or unknown errors are corrected in reoccurring versions. Errors in temperature estimates do vary over space meaning any gridded temperature value, even on higher quality maps, could still produce incorrect estimates. Check standard error (SE) maps to better understand spatial estimates of prediction error'
    dataStatement_val = dataStatement_val.format(date=date,mode=mode,island=county_list,inversion=str(meta['inversion']))
    
    #Set keyword field
    kw_list = ', '.join([county_list,'Hawaii',mode+' temperature prediction','daily temperature','temperature','climate','linear regression'])
    
    #Set credits and contacts
    credit_statement = 'All data produced by University of Hawaii at Manoa Dept. of Geography and the Enviroment, Ecohydology Lab in collaboration with the Water Resource Research Center (WRRC). Support for the Hawai‘i EPSCoR Program is provided by the Hawaii Emergency Management Agency.'
    contact_list = 'Keri Kodama (kodamak8@hawaii.edu), Matthew Lucas (mplucas@hawaii.edu), Ryan Longman (rlongman@hawaii.edu), Sayed Bateni (smbateni@hawaii.edu), Thomas Giambelluca (thomas@hawaii.edu)'
    
    #Arrange all meta fields and write to file
    field_value_list = {'attribute':'value','dataStatement':dataStatement_val,'keywords':kw_list,
        'county':island.lower(),'dataDate':date,'dataVersionType':'archival','tempStationFile':stn_file,'tempGridFile':temp_file,
        'tempSEGridFile':se_file,'crossValidationFile':cv_file,'GeoCoordUnits':'Decimal Degrees',
        'GeoCoordRefSystem':'+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0','XResolution':str(isl_dims['XResolution']),
        'YResolution':str(isl_dims['YResolution']),'ExtentXmin':str(isl_dims['Xmin']),
        'ExtentXmax':str(isl_dims['Xmax']),'ExtentYmin':str(isl_dims['Ymin']),
        'ExtentYmax':str(isl_dims['Ymax']),'stationCount':str(meta['nstn']),
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


# This code has been automatically covnerted to comply with the pep8 convention
# This the Linux command:
# $ autopep8 --in-place --aggressive  <filename>.py
if __name__ == '__main__':

    iCODE = 'BI'  # str(sys.argv[1])
    mode  = 'max' # str(sys.argv[2])
