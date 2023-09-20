# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 16:50:47 2023

@author: user
"""

import numpy as np
import pandas as pd


import scipy.io
from statsmodels.stats.anova import AnovaRM

#import seaborn.objects as so

#Participants labels with corresponding stimulation type for each session
Dict = {
        "AD0109": ['70 Hz','20 Hz','Sham'],
        "RP0129": ['70 Hz', 'Sham', '20 Hz'],
        "KM0404": ['20 Hz', 'Sham', '70 Hz'],
        "CD1107": ['Sham', '20 Hz', '70 Hz'],
        "FB0901": ['Sham', '70 Hz', '20 Hz'],
        "PB0526": ['20 Hz', 'Sham', '70 Hz'],
        "MB0522": ['20 Hz', '70 Hz', 'Sham'],
        "FB0210": ['70 Hz', '20 Hz', 'Sham'],
        "DF0720": ['20 Hz', '70 Hz', 'Sham'],
        "PT0728": ['20 Hz', 'Sham', '70 Hz'],
        "CB0724": ['Sham', '20 Hz', '70 Hz'],
        "JM0410": ['70 Hz', 'Sham', '20 Hz'],
        "GM0804": ['70 Hz', 'Sham', '20 Hz'],
        "TN0118": ['Sham', '20 Hz', '70 Hz'],
        "ME0212": ['Sham', '70 Hz', '20 Hz']     
        }

SubjectID = list(Dict)

#Loading mat files
ErrorMat = scipy.io.loadmat('ErrorMVC.mat')
ReactionTimeMat = scipy.io.loadmat('ReactionTimeMVC.mat')

#Extracting only the data of interest from mat files
Error = ErrorMat['Error']
ReactionTime = ReactionTimeMat['ReactionTime']

#Converting to dataframes
Errordf = pd.DataFrame (Error, columns = ['AD0109','RP0129', 'KM0404', 'FB0901', 'CD1107', 'PB0526', 'MB0522', 'FB0210', 'DF0720', 'PT0728', 'CB0724', 'JM0410', 'GM0804', 'TN0118', 'ME0212'])
RTdf = pd.DataFrame (ReactionTime, columns = ['AD0109', 'RP0129', 'KM0404', 'FB0901', 'CD1107', 'PB0526', 'MB0522', 'FB0210', 'DF0720', 'PT0728', 'CB0724', 'JM0410', 'GM0804', 'TN0118', 'ME0212'])

del ErrorMat
del ReactionTimeMat

Time = ['Baseline', 'tACS', '15min', '45min']
Sessions = ['S1', 'S2', 'S3']
fig = 1
time = [2,3,0,1]

TotalError = pd.DataFrame (columns = ['Error', 'Time', 'Subject', 'Stimulation'])
## LOOP FOR ERROR

#Loop for subjects
for u in range(14):
    sub = SubjectID[u+1]
    
    #Loop for sessions (1-3)
    for y in range(3):
        dfnew = pd.DataFrame(columns = ['Error', 'Time', 'Subject', 'Stimulation'])
            
        #Loop for Time (Base, tACS, 15min, 45min)    
        for i in range(4):
            
            #Extracting data to a new dataframe
            df = pd.DataFrame(columns = ['Error', 'Time'])
            df['Error'] = Errordf[sub][0][0,y][0,time[i]].tolist()[4:-1]
            df['Time']= Time[i]
            
            #Extracting values from lists to single floats
            indexes = df.index.values.tolist()
            
            for k in range(indexes[-1]+1):
                df['Error'][k] = df['Error'][k][0]
            
            
            #Conidtion that outliers that stand 3 SD from the mean
            condition = np.mean(df['Error'])+ (np.std(df['Error'])*3)
            
            #Removing outliers
            df2 = df.loc[df['Error'] < condition]
                
            #Concatenating all time dataframes into a single one for each session
            dfnew['Error']= df2['Error']
            dfnew['Time'] = Time[i]
            dfnew['Subject'] = sub
            dfnew['Stimulation'] = Dict[sub][y]
            TotalError = pd.concat([TotalError, dfnew], ignore_index=True)
            
print('Error')
anovares = AnovaRM(TotalError, 'Error', 'Subject', within = ['Time', 'Stimulation'], aggregate_func = 'mean').fit()
print(anovares)

TotalRT = pd.DataFrame (columns = ['Reaction Time', 'Time', 'Subject', 'Stimulation'])


##Reaction Time

for u in range(14):
    sub = SubjectID[u+1] 
    #Loop for sessions (1-3)
    for y in range(3):
        dfnew = pd.DataFrame(columns = ['Reaction Time', 'Time', 'Subject', 'Stimulation'])
            
        #Loop for Time (Base, tACS, 15min, 45min)    
        for i in range(4):
            #Extracting data to a new dataframe
            df = pd.DataFrame(columns = ['Reaction Time', 'Time'])
            df['Reaction Time'] = RTdf[sub][0][0,y][0,i].tolist()[4:-1]
            df['Time']= Time[i]
            
            #Extracting values from lists to single floats
            indexes = df.index.values.tolist()
            
            for k in range(indexes[-1]+1):
                df['Reaction Time'][k] = df['Reaction Time'][k][0]
            
            #Conidtion that outliers that stand 3 SD from the mean
            condition = np.mean(df['Reaction Time'])+ (np.std(df['Reaction Time'])*3)
            condition2 = np.mean(df['Reaction Time'])- (np.std(df['Reaction Time'])*3)
            #Removing outliers
            df22 = df.loc[df['Reaction Time'] < condition]
            df2 = df22.loc[df22['Reaction Time'] > condition2]
                
            #Concatenating all time dataframes into a single one for each session
            dfnew['Reaction Time']= df2['Reaction Time']
            dfnew['Time'] = Time[i]
            dfnew['Subject'] = sub
            dfnew['Stimulation'] = Dict[sub][y]
            TotalRT = pd.concat([TotalRT, dfnew], ignore_index=True)

print('Reaction Time')
anovares = AnovaRM(TotalRT, 'Reaction Time', 'Subject', within = ['Time', 'Stimulation'], aggregate_func = 'mean').fit()
print(anovares)








