"""   
File : data_O.py
Created by : Amen Ouannes
Brief : Python file for including all the functions designated to process ruuvi data and display 
"""


#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
from datetime import datetime
from dateutil import tz


# function to read the filenames and confert them to usable dataframes
def processing(filename):
    data = pd.read_csv(filename)
    data = ns_to_timestamp(data)
    data = data_drop(data)
    return data


#function to convert time column to readable time data timestamp
def ns_to_timestamp(data):
    #verification + converting values to int64
    data['time'] = data['time'].astype(np.int64)
    
    #changing time from ns to timestamp
    data['time'] = pd.to_datetime(data['time'], unit='ns')
    
    #converting to UTC-4
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/Montreal')
    data['time'] = data['time'].dt.tz_localize(from_zone).dt.tz_convert(to_zone)
    
    #strip my timestamp from the -4:00 indicator, no need to specify that
    data['time'] = data['time'].dt.tz_localize(None)
    
    return data 


#function for dropping useless columns
def data_drop(data):
    data.drop(['name', 'tags', 'sub_incr', 'prim_incr'], axis = 1, inplace = True)
    return data


#function to display the three axes at the same time        
def display(data):
    fig, (acc_x, acc_y, acc_z, state) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    acc_x.plot(data['time'], data['acc_x'], color = "red")
    
    acc_y.plot(data['time'], data['acc_y'], color = "green")
    
    acc_z.plot(data['time'], data['acc_z'], color = "blue")
    
    state.plot(data['time'], data['state'])
    acc_x.set_ylabel("acceleration on x axis", color = "red")
    acc_y.set_ylabel("acceleration on y axis", color = "green")
    acc_z.set_ylabel("acceleration on z axis", color = "blue")
    state.set_ylabel("state of the calf")
    plt.show()
    

#function to trunc values at the designated interval
#returns a time sequence of the dataframe from dt1 to dt2
def trunc(data, dt1, dt2):
    dt1 = datetime.strptime(dt1, "%Y-%m-%d %H:%M:%S.%f")
    dt2 = datetime.strptime(dt2, "%Y-%m-%d %H:%M:%S.%f")
    sample = data[(data['time'] >= dt1)
                            & (data['time'] <= dt2)]
    return sample


