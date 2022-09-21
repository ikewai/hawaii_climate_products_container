import sys
import pandas as pd
from datetime import datetime, timedelta
from monthly_state_temp import statewide_mosaic, create_tables

if __name__ == '__main__':
    if len(sys.argv) > 1:
        #If manual input of date
        date_str = sys.argv[1]
        date_time = pd.to_datetime(date_str)
        this_year = date_time.year
        this_mon = date_time.month
        month_st = datetime(this_year,this_mon,1)
        #Double check that the month is complete
        #If manual start is the same as today month start, month not complete
        today = datetime.today()
        today_year = today.year
        today_mon = today.month
        today_st = datetime(today_year,today_mon,1)
        if month_st == today_st:
            print('Month incomplete. Exiting.')
            quit()
    else:
        #Automatic real-time date set
        #Runs for the last complete month prior to today()
        today = datetime.today()
        this_year = today.year
        this_mon = today.month
        this_st = datetime(this_year,this_mon,1)
        prev = this_st - timedelta(days=1)
        prev_year = prev.year
        prev_mon = prev.month
        month_st = datetime(prev_year,prev_mon,1)
    
    month_date = month_st.strftime('%Y-%m-%d')
    #Tmax
    varname = 'Tmax'
    try: 
        mode = varname[1:]
        #Map
        statewide_mosaic(varname,month_date)
        #Std error
        statewide_mosaic(varname,month_date,'_se')
        create_tables(varname[0],mode,month_date)
        print(month_date,varname,'done')
    except:
        print(month_date,varname,'failed')
    #Tmin
    varname = 'Tmin'
    try: 
        mode = varname[1:]
        #Map
        statewide_mosaic(varname,month_date)
        #Std error
        statewide_mosaic(varname,month_date,'_se')
        create_tables(varname[0],mode,month_date)
        print(month_date,varname,'done')
    except:
        print(month_date,varname,'failed')

    #Tmean
    varname = 'Tmean'
    try: 
        mode = varname[1:]
        #Map
        statewide_mosaic(varname,month_date)
        #Std error
        statewide_mosaic(varname,month_date,'_se')
        create_tables(varname[0],mode,month_date)
        print(month_date,varname,'done')
    except:
        print(month_date,varname,'failed')
