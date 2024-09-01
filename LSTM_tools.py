"""
File : LSTM_tools.ipynb
Created by : Amen Ouannes
Brief : LSTM  functions to prepare the dataset, build and extract results of the built models

"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
#importing machine learning customized tools 
import ML_tools as tools
import importlib
importlib.reload(tools)
#tensorflow and sci-kit learn libraries methods to use for deep learning
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings
#prevent unnecesseary errors from the 2. and superior tensorflow versions 
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


#function to extract the most frequent class in an array
def most_frequent(arr):
    unique, counts = np.unique(arr, return_counts=True)
    return unique[np.argmax(counts)]


#function to transform dataset into lstm time sequences
def transform_X_y(data):
    """
     function to transform the dataset to numpys with the structures:
     X : (instances, timestep, features) for the input 
     y : (instances, label) for the annotations
    """
    data.index = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    data, label_encoder = tools.encode(data) #encode the dataset labels
    

    data_np = data.to_numpy()  # Convert to float32 explicitly
    X = []
    y = []
    
    
    for i in range(len(data_np) - 55):
        row = [j[1:10] for j in data_np[i:i+50]]
        X.append(row)
        
        annotations = data_np[i+50:i+55]  # The label is the data point after the window
        labels = [sequence[-1] for sequence in annotations] #regroup the labels on the most frequent
        y.append(most_frequent(labels))
        
        #y.append(data_np[i+50][-1])
        
    X = np.array(X)
    y = np.array(y)
    
    return X, y, label_encoder


#function to generate data for the train
def generate_train(calf_values, calf_labels):
    """
    Function to generate train and test from X and y given from the transform_X_y
    Train set size : 90%
    Validation set size : 5% 
    Test set size : 5%
    
    """
    X = []
    y = []
    #concatenate all the calves
    for key, value in calf_values.items():
        labels = calf_labels[key]
        
        X.append(value)
        y.append(calf_labels[key])

    # Convert lists to numpy arrays and reshape it
    X = np.array(X).astype(np.float32)
    instances = X.shape[0]*X.shape[1]
    X = X.reshape((instances , 50, 9)) #reshape from (5, n, 40, 4) to (5*n, 40, 4) to keep a linear structure
    y = np.array(y)
    y = y.reshape(instances, 1)   
    #one-hot encoding
    y = to_categorical(y, num_classes=len(np.unique(y)))
    #extract train data    
    X_train, X_other, y_train, y_other = train_test_split(X, y, test_size = 0.10, random_state = 581)
    #generate validation and test from the rest
    X_valid, X_test, y_valid, y_test = train_test_split(X_other, y_other, test_size = 0.50, random_state = 581) 
    return X_train, X_valid, X_test, y_train, y_valid, y_test, 


#function to generate sampled dataset for the model developpement
def generate_train_sampled(calf_values, calf_labels):
    """
    Function to generate train and test from X and y given from the transform_X_y 
    this functions keep the exact number of instances for each class (up, down , drink)
    Train set size : 90%
    Validation set size : 5% 
    Test set size : 5%
    
    """
    X = []
    y = []    
    #concatenate all the calves
    for key, value in calf_values.items():
        labels = calf_labels[key]

        #Append in the bigger dataset
        X.append(value)
        y.append(labels)
        
    # Convert lists to numpy arrays and reshape it
    X = np.array(X).astype(np.float32)
    instances = X.shape[0]*X.shape[1]
    X = X.reshape((instances, 5, 9)) #reshape from (5, n, 40, 4) to (5*n, 40, 4) to keep a linear structure
    y = np.array(y)
    y = y.reshape(instances, 1)
        
    mask_drink = (y == 1).flatten()
    mask_up = (y == 2).flatten()
    mask_down = (y == 0).flatten()
        
    # Extract data for each class
    drink_x = X[mask_drink]
      
    random_indices_u = np.random.choice(X[mask_up].shape[0], size=len(drink_x), replace=False) #generate random up indices with the equivalent size of drink
    random_indices_d = np.random.choice(X[mask_down].shape[0], size=len(drink_x), replace=False) #generate random down indices with the equivalent size of drink
        
    up_x = X[mask_up][random_indices_u] #select random elements from the up array with the size of drink
    down_x = X[mask_down][random_indices_d] #select random elements from the down array with the size of drink
        
    #Extract labels for each class
    drink_y = y[mask_drink]
    up_y = y[mask_up][random_indices_u]
    down_y = y[mask_down][random_indices_d]
    
    # Concatenate the separated data back together
    X_combined = np.concatenate((drink_x, up_x, down_x))
    y_combined = np.concatenate((drink_y, up_y, down_y))
    
    #one-hot encoding
    y_combined = to_categorical(y_combined, num_classes=len(np.unique(y)))
    #extract train data    
    X_train, X_other, y_train, y_other = train_test_split(X_combined, y_combined, test_size = 0.10, random_state = 581)
    #generate validation and test from the rest
    X_valid, X_test, y_valid, y_test = train_test_split(X_other, y_other, test_size = 0.50, random_state = 581) 
    return X_train, X_valid, X_test, y_train, y_valid, y_test


#function to generate train, validation, test set 
def leave_one_calf_out(calf_values, calf_labels, id_, features=4):
    """
    Function to generate train and test from X and y given from the transform_X_y 
    this functions keep a single calf completely out of the training set and then prepare his data for validation and test sets.
    Train set size : 80%
    Validation set size : 15%
    Test set size : 5%
    
    """
    y_train = []
    X_train = []
    #extract calves for the train
    for key, value in calf_values.items():
        if key != id_:
            X_train.append(value)
            y_train.append(calf_labels[key])
            
    # Convert lists to numpy arrays and reshape it
    X_train = np.array(X_train).astype(np.float32)
    instances = X_train.shape[0]*X_train.shape[1]
    X_train = X_train.reshape((instances, 50, features))  #reshape
    
    y_train = np.array(y_train)#.astype(np.float32)
    y_train = y_train.reshape(instances,1)        #reshape from (None, m, 1) to (m, 1)

    # Get the last calf and reshape its data
    X_id = np.array(calf_values[id_]).astype(np.float32)
    X_id = X_id.reshape((X_id.shape[0], 50, features))
    y_id = np.array(calf_labels[id_]).astype(np.float32)
    y_id = y_id.reshape((X_id.shape[0], 1))
    #reshape my data to binary array representing the class
    y_train = to_categorical(y_train, num_classes=3)
    y_id = to_categorical(y_id, num_classes=3) 
    #do a validation-test split
    X_valid, X_test, y_valid, y_test = train_test_split(X_id, y_id, test_size = 0.25, random_state = 581)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


#function to calculate performances on the train , test and validation sets 
def performances(model, X_train, y_train, X_valid, y_valid, X_test, y_test):
    #evaluate models
    #train_loss, train_accuracy = model.evaluate(X_train, y_train)
    valid_loss, valid_accuracy = model.evaluate(X_valid, y_valid)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    #print(f"train set performance = {train_accuracy*100:.3f}%")
    print(f"validation set performance = {valid_accuracy*100:.3f}%")
    print(f"test set performance = {test_accuracy*100:.3f}%")    

    
#function to draw the confusion matrix for the lstm models     
def confusion_lstm(model, X, y):
    predictions = model.predict(X) #generate predictions
    
    y = np.argmax(y, axis=1)  
    predictions = np.argmax(predictions, axis=1)  
    
    # Generate the confusion matrix
    cm = confusion_matrix(y, predictions)
    cm_normalized = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100
    #plot with seaborn
    sns.heatmap(cm_normalized, annot=True, xticklabels=["down", "drink", "up"],
                yticklabels= ["down", "drink", "up"], fmt='.1f', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix in percentages(recall)')
    plt.show()
    
    
#function to generate a dataframe of predictions and annotations
def predictions(model, X, y):
    predictions = model.predict(X) #generate predictions
    df_pred = pd.DataFrame(predictions)
    binary_df = df_pred.apply(tools.convert_to_binary, axis=1) #convert to a single column binary array of prediction
    y = pd.DataFrame(y).apply(tools.convert_to_binary, axis=1) 
    results = pd.concat([binary_df, y], axis =1, ignore_index=True)
    
    return results
