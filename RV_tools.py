"""   
File : RV_tools.py
Created by : Amen Ouannes
Brief : Python file for including all the functions designated to process training data and features
"""
#import libraries
import pandas as pd
from scipy.signal import butter, filtfilt
import statistics as stat
import multiprocessing
import math as math
import numpy as np
import data_O as tool
pd.options.mode.chained_assignment = None


#function to classify data per calf
def classify(data, calves):
    for name, df in calves.items():
        #sampling data for each calf one at a time 
        sample = data[data['mac'] == name]
        a = len(df)
        #adding to each dataframe
        df = pd.concat([df, sample], ignore_index=True)
        calves[name] = df
        #verify if the concatenation did really happen
        assert(len(df) == a + len(sample))
    return calves

# function to calculate mean value on a certain interval
def create_mean(data, axis):
    mean_axis = data.groupby('time')[axis].mean()
    df = mean_axis.reset_index()
    df.columns = ['time', axis] 
    return df

# function to calculate standard deviation value on a certain interval 
def create_std(data, axis):
    dev_ = 'st_dev_' + axis[-1]
    st_dev = data.groupby('time')[axis].std()
    st_dev = st_dev.fillna(0)
    df = st_dev.reset_index()
    df.columns = ['time', dev_]  
    return df

# funtion to generate features from accelerations
def features_generator(data):
    #apply mean value of x and its standard deviation for each one second interval
    mean_x = create_mean(data, 'acc_x')
    st_dev_x = create_std(data, 'acc_x')
    #var_x = variance(data,'acc_x')
    #mad_x = mad(data, 'acc_x')
    #irq_x = irq_range(data, 'acc_x')
    #apply mean value of y  and its standard deviation for each one second interval
    mean_y = create_mean(data, 'acc_y')
    st_dev_y = create_std(data, 'acc_y')
    #var_y = variance(data,'acc_y')
    #mad_y = mad(data, 'acc_y')
    #irq_y = irq_range(data, 'acc_y')
    
    #apply mean value of z and its standard deviation for each one second interval
    mean_z = create_mean(data, 'acc_z')
    st_dev_z = create_std(data, 'acc_z')
    #var_z = variance(data,'acc_z')
    #mad_z = mad(data, 'acc_z')
    #irq_z = irq_range(data, 'acc_z')
    
    #merge all of it to obtain the base of features before computations
    #dataframes = [mean_x, mean_y, mean_z, st_dev_x, st_dev_y, st_dev_z, 
    #              var_x, var_y, var_z, mad_x, mad_y, mad_z, irq_x, irq_y, irq_z]
    
    
    
    dataframes = [mean_x, mean_y, mean_z, st_dev_x, st_dev_y, st_dev_z] 
    features = dataframes[0]
    
    # Loop through the remaining DataFrames and merge them sequentially
    for df in dataframes[1:]:
        features = pd.merge(features, df, on='time', how='outer')

    #add Amag value
    features['Amag'] = np.sqrt(features['acc_x']**2 + features['acc_y']**2 + features['acc_z']**2)
    
    #add dynamic accelerations
    features['ax_dynamic'] = features['acc_x'] - features['acc_x'].mean()
    features['ay_dynamic'] = features['acc_y'] - features['acc_y'].mean()
    features['az_dynamic'] = features['acc_z'] - features['acc_z'].mean()

    #add static accelerations
    features['static_acc_x'] = low_pass_filter(features['acc_x'])
    features['static_acc_y'] = low_pass_filter(features['acc_y'])
    features['static_acc_z'] = low_pass_filter(features['acc_z'])
    
    #calculate OBDA values
    features['OBDA'] = np.sqrt(features['ax_dynamic']**2 + features['ay_dynamic']**2 + features['az_dynamic']**2)
    
    #calculate VeBDA values
    features['VeDBA'] = np.abs(features['ax_dynamic']) + np.abs(features['ay_dynamic']) + np.abs(features['az_dynamic'])
    
    #calculate roll values
    features['roll'] = np.arctan(features['acc_y']/features['acc_z'])
    
    #calculate pitch values
    features['pitch'] = np.arctan(-features['acc_x']/np.sqrt(features['acc_z']**2 + features['acc_y']**2))
    
  
    return features

#filtering function, keep only one value per second
def filter(data1):
    #for copyWarning reasons..
    data = data1
    #trunc values for the study span only
    data = tool.trunc(data, "2024-05-04 00:00:00.0", "2024-05-11 00:00:00.0")
    mac = data.iloc[2]['mac']
    # Group by 'time' and round up to the second
    data.loc[:, 'time'] = data['time'].dt.round('1s')
    #create new data frame based on features
    data['mac'] = mac
    features = features_generator(data) 
    features['mac'] = mac
    return features




###################################################################################
#function to regroup by minutes for lstm dataset
def regroup_min(df, minutes):
    # Convert the timestamp to datetime format
    df['time'] = pd.to_datetime(df['time'])

    # Set the timestamp as the index
    df.set_index('time', inplace=True)

    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
    regroup_int = minutes*60
    regroup_str = str(regroup_int) + 's'
    # Resample the numeric columns by the minute
    resampled_numeric = df[numeric_cols].resample(regroup_str).mean()

    # Handle non-numeric columns: take the first entry of each minute for the 'device' column
    resampled_non_numeric = df[non_numeric_cols].resample(regroup_str).first()

    # Combine the resampled numeric and non-numeric data
    resampled_df = pd.concat([resampled_numeric, resampled_non_numeric], axis=1)

    # Reset the index if needed
    resampled_df.reset_index(inplace=True)
    
    return resampled_df

#################################################################################
# function to calculate the variance value on a certain interval
def variance(data, axis):
    var_ = data.groupby('time')[axis].var()
    df = var_.reset_index()
    # Naming the new column explicitly to reflect it is variance of the axis
    df.columns = ['time', 'var_' + axis[-1]]
    return df
# function to calculate the interquartile range
def irq_range(data, axis):
    iq1 = data.groupby('time')[axis].quantile(0.25)
    iq3 = data.groupby('time')[axis].quantile(0.75)
    df = iq3-iq1
    df = df.reset_index()
    df.columns = ['time', 'iqr_' + axis[-1]]
    
    return df

# function that calculates the mean absolute deviation on a certain interval
def mad(data, axis):
    group = data.copy()
    grouped = group.groupby('time')[axis]
    mean = grouped.transform('mean')  # Use transform to align means with the original DataFrame size

    # Calculate absolute deviations from the mean for each group
    abs_deviation = np.abs(data[axis] - mean)

    # Group by 'time' again to calculate mean of these absolute deviations
    mad = abs_deviation.groupby(data['time']).mean()
    mad = mad.to_frame()
    mad = mad.rename(columns = {axis : 'mad_'+axis[-1]})
    return mad

# Define a low-pass filter function using the adjusted parameters
def low_pass_filter(data, cutoff=0.25, fs=1, order=4):  
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data
