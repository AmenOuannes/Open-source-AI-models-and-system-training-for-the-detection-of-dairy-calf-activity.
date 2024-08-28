"""
File : ML_tools.py
Created by : Amen Ouannes
Brief : Machine learning functions to visualize and interpret results from LSTM and RF 

"""

#importing libraries
#scikit-learn libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import r2_score
import xgboost as xgb
from xgboost import XGBClassifier
#deep learning libraries
import LSTM_tools as lstm
from tensorflow.keras.utils import to_categorical
#data science libraries
from datetime import datetime
import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#prevent minor error messages 
pd.set_option('future.no_silent_downcasting', True)

#----------------------------------------------------------------
# Ensemble learning models
# Set of functions useful for the RF and xgb training

# function to calculate performance
def performance(model, X_train, X_test, y_train, y_test):
    
    print(f"training performance = {model.score(X_train, y_train)*100:.3f}%")
    print(f"test performance = {model.score(X_test, y_test)*100:.3f}%")

    
#function to display a confusion matrix
def confusion(label_encoder, improved_Grid,X_test,y_test):
    y_pred = improved_Grid.predict(X_test) #save predicted values
    cm = sk_confusion_matrix(y_test, y_pred) #create confusion matrix
    
    cm_normalized = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100 #switch to percentages
    
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=label_encoder.classes_) #create a display

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))  # This also defines 'fig' properly
    cmd.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix in percentage')
    plt.show()

    
#function to encode the state into numeric classes
#returns the encoded dataset with the label encoder(for future use)
def encode(dataset):
    label_encoder = LabelEncoder()
    dataset['state_encoded'] = label_encoder.fit_transform(dataset['state'].astype(str))
    return dataset, label_encoder


#function that generates proper train and test data
#returns the train and test sets and labels
def generate_train(dataset, f_list):
    #divide the features from the real values
    X = dataset[f_list]
    y = dataset['state_encoded']
    #perform the train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 581)
    return X_train, X_test, y_train, y_test


#function that generates the parameter grid for random forest model
#returns a parameter grid of the hyperparameters
def hypers_rf(f_list):
    n_estimators = np.arange(100,201, 10) # Number of trees
    max_depth = np.arange(3,10,2) # Number of features to consider
    
    min_samples_split = np.arange(2,10,2) # Minimum number of samples to split a node
    min_samples_leaf = np.arange(1, 6, 1) # Minimum number of samples required to be at a leaf node
    
    max_features = ['log2', 'sqrt', 0.5] #number of features per tree
    bootstrap = [False, True] #bootstrap
    max_leaf_nodes = None # max leaf nodes
    criterion = ['gini', 'entropy'] # criterion
    
    # Create a parameter grid
    param_grid = {'n_estimators' : n_estimators,
                   'max_depth' : max_depth,
                   'max_features' : max_features,
                   'bootstrap' : bootstrap,
                  'min_samples_split': min_samples_split,
                   #'max_leaf_nodes' : max_leaf_nodes,
                   'criterion' : criterion         }
    return param_grid


#function that generates the parameters grid for the extreme gradient boosting model
#returns a parameter grid of the hyperparameters
def hypers_xgb():
    
    param_grid = {
    'booster' : ['gbtree'],
    'learning_rate' : np.arange(0.05, 0.35, 0.1),
    'n_estimators' : np.arange(100, 200, 25),
    'max_depth' : np.arange(3, 10, 3),
    'min_child_weight' : np.arange(1, 5, 2),
    'gamma' : np.arange(0.1,0.3,0.1),
    'subsample' : np.arange(0.5, 1, 0.2),
    'reg_alpha' : [0, 0.01, 0.1],
    'colsample_bytree' :[0.5, 0.7, 1.0]
    }
    return param_grid
#-------------------------------------------------------------------------------
#functions to interpret results from the Ensemble learning


#function to read results from pickles
#returns a dataframe of performances of each used model and the features importance vectors
def read_pk(filename):
    with open(filename, 'rb') as f:
        frame = pk.load(f)
        dist  = pk.load(f)
    return frame, dist


#function to convert probabilites into real values by taking the max value    
def convert_to_binary(row):
    max_idx = row.idxmax()
    binary_row = np.zeros_like(row)
    binary_row[max_idx] = 1
    return binary_row


#function to compare between train and test results
def train_vs_test(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    indices = range(1,len(df)+1)
    # Width of each bar
    width = 0.35
    # Plotting train scores
    ax.bar([i - width/2 for i in indices], df['train'], width=width, label='Train', color='blue')
    # Plotting test scores
    ax.bar([i + width/2 for i in indices], df['test'], width=width, label='Test', color='red')
    ax.set_xticks(indices)
    ax.set_xticklabels(['Model {}'.format(i) for i in indices])
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Train and Test Scores for each training')
    ax.legend()
    plt.show()
    

#function to generate the list of macs
def list_id():
    return ["D1:B6:FC:34:99:3A", "D0:C9:50:5E:88:F8", "F4:CC:E1:C5:8A:A9", "FB:E5:24:5B:8A:68", "D3:6B:03:69:94:4E"]

#function to generate the dictionnary of calf numbers and macs
def macs():
    return {"D1:B6:FC:34:99:3A" : 7480, "F4:CC:E1:C5:8A:A9" :8855, "FB:E5:24:5B:8A:68" : 8851, "D0:C9:50:5E:88:F8" :8846, "D3:6B:03:69:94:4E" : 8854}

#important features list
f_list = ['acc_x', 'acc_y', 'acc_z','st_dev_x', 'st_dev_y', 'st_dev_z', 'ax_dynamic', 'ay_dynamic', 'az_dynamic']


#function to trunc values at the designated interval 
#returns a dataframe for the designed time period
def trunc(data, dt1, dt2):
    dt1 = datetime.strptime(dt1, "%Y-%m-%d %H:%M:%S")
    dt2 = datetime.strptime(dt2, "%Y-%m-%d %H:%M:%S")
    sample = data[(data['time'] >= dt1)
                            & (data['time'] <= dt2)]
    return sample


#function to draw a graph of the predicted versus actual labels for the RF model
def compare_rf(real, calf, rf_model,label_encoder, dt1, dt2, required_columns = f_list):
    sequence = trunc(real[calf], dt1, dt2)
    sequence = sequence[5:] #ensure there's no overlap between lstm and rf(first prediction for lstm is after 5 seconds).

    labels = sequence['state'] #extract the annotations
    values = sequence[required_columns] #prepare the test set
    y_pred = rf_model.predict(values) #pedict the labels
    
    probabilities = rf_model.predict_proba(values) #predict probabilities
    y_pred_mapped = label_encoder.inverse_transform(y_pred) #transform from [0,1,2] to [down, drink, up]
    
    plt.figure(figsize=(12, 6))  
    plt.plot(sequence['time'], labels, label='Actual Labels')    # Plot predicted values
    plt.plot(sequence['time'], y_pred_mapped, label='Predicted Values (RF)', linestyle='dashed')
    
    # Set plot title and labels
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.legend() 
    plt.show()      
    #plot for class probabilities
    plt.figure(figsize=(12, 6)) 
    plt.plot(sequence['time'], [probabilities[i][0]for i in range(len(probabilities))], label='down', linestyle='solid') 
    plt.plot(sequence['time'], [probabilities[i][1]for i in range(len(probabilities))], label='drink', linestyle='solid') 
    plt.plot(sequence['time'], [probabilities[i][2]for i in range(len(probabilities))], label='up', linestyle='solid') 
    plt.legend()    
    plt.show()


#function to draw a graph of the predicted versus actual labels for the RF model    
def compare_lstm(real, calf, lstm_model, dt1, dt2, features = 4):
    sequence = trunc(real[calf], dt1, dt2)
    # Extract values and labels
    values_lstm, labels_lstm, label_encoder = lstm.transform_X_y(sequence)
    
    # Convert to NumPy array and ensure correct dtype
    values_lstm = np.array(values_lstm, dtype=np.float32)
    values_lstm = values_lstm.reshape((values_lstm.shape[0], 5, features))  # Adjust shape based on your data
    labels_lstm = np.array(labels_lstm)
    labels_lstm = labels_lstm.reshape((labels_lstm.shape[0],))  # Adjust shape based on your data
    labels_lstm = to_categorical(labels_lstm, num_classes=3)

    # Make predictions
    lstm_probabilities = lstm_model.predict(values_lstm)
    lstm_y_pred = np.argmax(lstm_probabilities, axis=1)
    # Create a mapping dictionary
    mapping = {1: 'drink', 0: 'down', 2: 'up'}

    # Use np.vectorize to apply the mapping to the entire array
    vectorized_mapping = np.vectorize(mapping.get)
    lstm_y_pred_mapped = vectorized_mapping(lstm_y_pred)
    sequence = sequence[5::] #same adjustement done with the rf graph
        
    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(sequence['time'], sequence['state'], label='Actual Labels')
    plt.plot(sequence['time'], lstm_y_pred_mapped, label='Predicted Values (LSTM)', linestyle='dashed')

    plt.title('Actual vs Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.legend()  # Add legend to the plot
    plt.show()  # Show the plot

    # Plot class probabilities
    plt.figure(figsize=(12, 6))
    plt.plot(sequence['time'], lstm_probabilities[:, 0], label='down (LSTM)', linestyle='dashed')
    plt.plot(sequence['time'], lstm_probabilities[:, 1], label='drink (LSTM)', linestyle='dashed')
    plt.plot(sequence['time'], lstm_probabilities[:, 2], label='up (LSTM)', linestyle='dashed')

    plt.title('Class Probabilities')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.legend()  # Add legend to the plot
    plt.show()  # Show the plot
    


#------------------------------------------------------------------------
#functions to do a full day presentation with n minutes precision



#function to regroup data on n minutes with selecting the most recurrent class on those minutes
#returns a regrouped dataset on n minutes
def regroup_min(df, minutes):
    # Convert the timestamp to datetime format
    df['time'] = pd.to_datetime(df['time'])
    # Set the timestamp as the index
    df.set_index('time', inplace=True)
    regroup_int = minutes * 60
    regroup_str = str(regroup_int) + 's'
    
    def most_frequent(series):
        return series.mode().iloc[0] if not series.mode().empty else None
    
    resampled_df = df.select_dtypes(exclude='number').resample(regroup_str).apply(most_frequent)
    # Reset the index 
    resampled_df.reset_index(inplace=True)
    return resampled_df


#function to generate an array of prediction regrouped per minutes
#returns a dataframe with actual labels and predictions
def generate_predictions(calf,lstm_model, dt1_step, dt1, dt2,dt_annot, minutes):
    sequence = trunc(calf, dt1_step, dt_annot)
    # Extract values and labels
    values_lstm, labels_lstm, label_encoder = lstm.transform_X_y(sequence)
    mapping = {1: 'drink', 0: 'down', 2: 'up'}
    
    # Convert to NumPy array and ensure correct dtype
    values_lstm = np.array(values_lstm, dtype=np.float32)
    values_lstm = values_lstm.reshape((values_lstm.shape[0], 40, 4))  # Adjust shape based on your data

    labels_lstm = np.array(labels_lstm)
    labels_lstm = labels_lstm.reshape((labels_lstm.shape[0],))  # Adjust shape based on your data
    labels_lstm = to_categorical(labels_lstm, num_classes=3)

    # Make predictions
    lstm_probabilities = lstm_model.predict(values_lstm)
    lstm_y_pred = np.argmax(lstm_probabilities, axis=1)
    vectorized_mapping = np.vectorize(mapping.get)
    lstm_y_pred_mapped = vectorized_mapping(lstm_y_pred)
    
    
    day = trunc(calf, dt1, dt2)
    day = day[['time','state']]
    
    lstm.confusion_lstm(lstm_model, values_lstm, labels_lstm)
    print(lstm_model.evaluate (values_lstm, labels_lstm))
    lstm_df = pd.DataFrame(lstm_y_pred_mapped, columns=['Predictions'])
    day = pd.DataFrame(day, columns=['time','state'])
    day = day.reset_index(drop=True)
    lstm_df = lstm_df.reset_index(drop=True)
    lstm_df['time'] = day['time']
    print(len(day), len(lstm_df))
    assert(len(day)==len(lstm_df))
    undersampled = pd.merge(day, lstm_df, on='time', how='inner')
    undersampled = regroup_min(undersampled, minutes)
    return undersampled


#function to draw the predictions vs true labels
def visualize(day):
    plt.figure(figsize=(12, 6))
    plt.plot(day['time'], day['state'], label='Actual Labels')
    plt.plot(day['time'], day['Predictions'], label='Predicted Values (LSTM)', linestyle='dashed')

    plt.title('Actual vs Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.legend()  # Add legend to the plot
    plt.show()  # Show the plot
