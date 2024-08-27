"""
    File : AX3_tools.py
    Created by : Amen Ouannes
    Brief : Python file for including all the functions designated to process AX3 data
"""
from datetime import datetime, timedelta
import pandas as pd
import pickle as pk
from dateutil import tz
pd.set_option('future.no_silent_downcasting', True)

#time trunc function
def trunc(data, dt1, dt2):
    dt1 = datetime.strptime(dt1, "%Y-%m-%d %H:%M:%S")
    dt2 = datetime.strptime(dt2, "%Y-%m-%d %H:%M:%S")
    sample = data[(data['time'] >= dt1)
                 & (data['time'] < dt2)]
    return sample

#function that prepares the dataframe for use
def apply(data, num):
    data['calfNumber'] = num
    data = data.drop(['time', 'temp', 'battery', 'light','x', 'y', 'z'], axis = 1)
    data = data.rename(columns={'date.time.EST': 'time', 'up.down.hmm' : 'state'})

    data['time'] = pd.to_datetime(data['time'])
    data = trunc(data, "2024-05-04 00:00:00", "2024-05-11 00:00:01")
    return data

#Regroup by 5 seconds
def regroup(data1):
    # Convert 'time' column to datetime
    data = data1
    data['time'] = pd.to_datetime(data['time'])
    # Set 'time' as the index
    data.set_index('time', inplace=True)
    # Resample the DataFrame by n-second intervals and calculate the mean
    resampled_data = data.resample('1s').max()
    # Reset the index
    resampled_data.reset_index(inplace=True)
    resampled_data['state'].astype(int)
    return resampled_data



#functions for the feather files, precise movement per second
def treat(data):
    data = data.to_pandas()
    
    data['time'] = pd.to_datetime(data['time'], unit='s')
    #converting to UTC-4
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/Montreal')
    data['time'] = data['time'].dt.tz_localize(from_zone).dt.tz_convert(to_zone)
    
    #strip my timestamp from the -4:00 indicator, no need to specify that
    data['time'] = data['time'].dt.tz_localize(None)
    data.rename(columns={'up.down.hmm': 'state'}, inplace=True)
    return data

def download(data, target_pkl):
    with open(target_pkl, 'wb')as f:
        pk.dump(data, f)