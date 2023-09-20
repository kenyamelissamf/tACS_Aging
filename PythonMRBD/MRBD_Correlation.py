# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 19:08:19 2023

@author: Kenya
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-colorblind')

import scipy.io
import scipy.stats

#import seaborn.objects as so

#Participants labels with corresponding stimulation type for each session and their age
Dict = {
        "AD0109": ['70 Hz','20 Hz','Sham', 69.7],
        "RP0129": ['70 Hz', 'Sham', '20 Hz', 76.6],
        "KM0404": ['20 Hz', 'Sham', '70 Hz',67.5],
        "CD1107": ['Sham', '20 Hz', '70 Hz', 66],
        "FB0901": ['Sham', '70 Hz', '20 Hz', 67.1],
        "PB0526": ['20 Hz', 'Sham', '70 Hz', 66.5],
        "MB0522": ['20 Hz', '70 Hz', 'Sham', 75.4],
        "FB0210": ['70 Hz', '20 Hz', 'Sham', 66.8],
        "DF0720": ['20 Hz', '70 Hz', 'Sham', 66.5],
        "PT0728": ['20 Hz', 'Sham', '70 Hz', 71.5],
        "CB0724": ['Sham', '20 Hz', '70 Hz', 66.6],
        "JM0410": ['70 Hz', 'Sham', '20 Hz', 65.9],
        "GM0804": ['70 Hz', 'Sham', '20 Hz', 78.6],
        "TN0118": ['Sham', '20 Hz', '70 Hz', 72.2],
        "ME0212": ['Sham', '70 Hz', '20 Hz', 69.2]     
        }

SubjectID = list(Dict)
age = [69.7, 76.6, 67.5, 66, 67.1, 66.5, 75.4, 66.8, 66.5, 71.5, 66.6, 65.9, 78.6, 72.2, 69.2]

Channels = ['FC1', 'C3', 'CP5', 'CP1', 'FC3', 'C5', 'C1', 'CP3']
Topography = ['FC3', 'FC1', 'C5', 'C3', 'C1', 'CP5', 'CP3', 'CP1']
time = ['Baseline', 'Post-15min', 'Post-45min']
Sessions = ['S1', 'S2', 'S3']
fig = 0

Baseline_Total= pd.DataFrame(columns = Channels)
MRBD_Total = pd.DataFrame(columns = Channels)

#Loop for subjects
for s in range(15):
    MeanMRBDs = []
    sub = SubjectID[s]
    ses = 0
    print(sub + ' ' + Dict[sub][ses])
    #Loading mat files
    BaselineMat = scipy.io.loadmat('MeanMRBD_' + sub +'_' + Sessions[ses] + '_Baseline.mat')
    
    Baseline = pd.DataFrame()
    
    for chan in range(8):
        Baseline[Channels[chan]] = BaselineMat['Value'][chan:len(BaselineMat['Value']):8,1]
   
        #Remove outliers
        
        #Conidtion that outliers that stand 3 SD from the mean
        conditionbase = np.mean(Baseline[Channels[chan]])+ (np.std(Baseline[Channels[chan]])*3)
 
        #Printing which outliers were removed
        for k in range(len(Baseline)):
            if Baseline[Channels[chan]][k] > conditionbase:
                print('Error Removed outlier: Trial #' + str(k+5) + ' Value: ' + str(Baseline[Channels[chan]][k]) + ' ' + sub + ' ' + Dict[sub][ses] + ' Baseline' )
        
        Baseline[Channels[chan]] = Baseline[Channels[chan]].loc[Baseline[Channels[chan]] < conditionbase]
        MRBD = Baseline[Channels[chan]].mean()
        MeanMRBDs.append(MRBD)
    
    MRBD_Total.loc[sub] = MeanMRBDs
plt.figure(fig, figsize= (15,15))
subp = 331
fig = fig+1

# Correlation with age      
print ('Correlation with age')
for chan in range(8):
    print (Topography[chan])
    MRBDS = MRBD_Total[Topography[chan]]
    slope, intercept, r, p, stderr = scipy.stats.linregress(MRBDS, age)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}, p={p:.3f}'
    fig, ax = plt.subplots(subp+chan+1)
    ax.plot(MRBDS, age, linewidth=0, marker='s', label='Data points')
    ax.plot(MRBDS, intercept + slope * MRBDS, label=line)
    ax.set_xlabel('MRBD (%)')
    ax.set_ylabel('Age (years)')
    plt.title(label = 'Electrode: ' + Topography[chan])
    plt.xlim([-60,42])
    legend = ax.legend(frameon = 1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    ax.grid(color = 'grey')
    plt.show()

# Correlation with motor performance
print ('Correlation with motor performance')      

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

Error_Total = pd.DataFrame(columns = ['Error'])
#Loop for subjects
for u in range(15):
    MeanError = []
    sub = SubjectID[u] 
    dfall2 = pd.DataFrame (columns = ['Values'])
    #y = Session1
    y = 0
    dfall = pd.DataFrame (columns = ['Values'])
    
    #i = Baseline
    i = 0
    #Extracting data to a new dataframe
    df1 = pd.DataFrame(columns = ['Values'])
    df1['Values'] = Errordf[sub][0][0,y][0,time[i]].tolist()[4:-1]
    indexes = df1.index.values.tolist()
    
    for k in range(indexes[-1]+1):
        df1['Values'][k] = df1['Values'][k][0]
        
    condition = np.mean(df1['Values'])+ (np.std(df1['Values'])*3)
    dfout = df1.loc[df1['Values'] < condition]
    dfall = pd.concat([dfall,dfout], ignore_index=True)
    del df1
    dfall2 = pd.concat([dfall2,dfall], ignore_index=True)
    del dfall
    print (sub + ' Error Max: '+ str(dfall2['Values'].max()))
    ymax = dfall2['Values'].max()
    ymin = dfall2['Values'].min()
    
    y = 0
    dfnew = pd.DataFrame(columns = ['Error', 'Time'])
    i = 0    
        
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
    
    #Printing which outliers were removed
    for k in range(indexes[-1]+1):
        if df['Error'][k] > condition:
            print('Error Removed outlier: Trial #' + str(k+5) + ' Value: ' + str(df['Error'][k]) + ' ' + sub + ' ' + Dict[sub][y] + ' ' + Time[i] )
    
    #Removing outliers
    df2 = df.loc[df['Error'] < condition]
        
    #Concatenating all time dataframes into a single one for each session
    dfnew = pd.concat([dfnew,df2], ignore_index=True)
    del df
        
    
    #Converting to numpy array
    dfnew['Error'] = np.array(dfnew['Error'], dtype=np.float64)
    
    MeanError = dfnew['Error'].mean()
    
    Error_Total.loc[sub] = MeanError
    
#Accuracy correlation with Age
ERRORS = Error_Total['Error']
slope, intercept, r, p, stderr = scipy.stats.linregress(ERRORS, age)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}, p={p:.3f}'
fig, ax = plt.subplots()
ax.plot(ERRORS, age, linewidth=0, marker='s', label='Data points')
ax.plot(ERRORS, intercept + slope * ERRORS, label=line)
ax.set_xlabel('Error')
ax.set_ylabel('Age (years)')
plt.title(label = 'Motor task accuracy and age')

legend = ax.legend(frameon = 1)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
ax.grid(color = 'grey')
plt.show()
            

#Error correlation with MRBD    
print ('MRBD correlation with motor performance')
for chan in range(8):
    print (Topography[chan])
    MRBDS = MRBD_Total[Topography[chan]]
    slope, intercept, r, p, stderr = scipy.stats.linregress(MRBDS, ERRORS)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}, p={p:.3f}'
    fig, ax = plt.subplots()
    ax.plot(MRBDS, ERRORS, linewidth=0, marker='s', label='Data points')
    ax.plot(MRBDS, intercept + slope * MRBDS, label=line)
    ax.set_xlabel('MRBD(%)')
    ax.set_ylabel('Error')
    plt.xlim([-60,42])
    plt.title(label = 'Motor task accuracy and MRBD at electrode ' + Topography[chan])
    
    legend = ax.legend(frameon = 1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    ax.grid(color = 'grey')
    plt.show()


#Reaction Time
RT_Total = pd.DataFrame(columns = ['Reaction Time'])
#Loop for subjects
for u in range(15):
    MeanError = []
    sub = SubjectID[u] 
    dfall2 = pd.DataFrame (columns = ['Values'])
    #y = Session1
    y = 0
    dfall = pd.DataFrame (columns = ['Values'])
    
    #i = Baseline
    i = 0
    #Extracting data to a new dataframe
    df1 = pd.DataFrame(columns = ['Values'])
    df1['Values'] = RTdf[sub][0][0,y][0,time[i]].tolist()[4:-1]
    indexes = df1.index.values.tolist()
    
    for k in range(indexes[-1]+1):
        df1['Values'][k] = df1['Values'][k][0]
        
    condition = np.mean(df1['Values'])+ (np.std(df1['Values'])*3)
    dfout = df1.loc[df1['Values'] < condition]
    dfall = pd.concat([dfall,dfout], ignore_index=True)
    del df1
    dfall2 = pd.concat([dfall2,dfall], ignore_index=True)
    del dfall
    print (sub + ' Reaction Time Max: '+ str(dfall2['Values'].max()))
    ymax = dfall2['Values'].max()
    ymin = dfall2['Values'].min()
    
    y = 0
    dfnew = pd.DataFrame(columns = ['Reaction Time', 'Time'])
    i = 0    
        
    #Extracting data to a new dataframe
    df = pd.DataFrame(columns = ['Error', 'Time'])
    df['Reaction Time'] = RTdf[sub][0][0,y][0,time[i]].tolist()[4:-1]
    df['Time']= Time[i]
    
    #Extracting values from lists to single floats
    indexes = df.index.values.tolist()
    
    for k in range(indexes[-1]+1):
        df['Reaction Time'][k] = df['Reaction Time'][k][0]
    
    
    #Conidtion that outliers that stand 3 SD from the mean
    condition = np.mean(df['Reaction Time'])+ (np.std(df['Reaction Time'])*3)
    
    #Printing which outliers were removed
    for k in range(indexes[-1]+1):
        if df['Reaction Time'][k] > condition:
            print('Reaction Time Removed outlier: Trial #' + str(k+5) + ' Value: ' + str(df['Reaction Time'][k]) + ' ' + sub + ' ' + Dict[sub][y] + ' ' + Time[i] )
    
    #Removing outliers
    df2 = df.loc[df['Reaction Time'] < condition]
        
    #Concatenating all time dataframes into a single one for each session
    dfnew = pd.concat([dfnew,df2], ignore_index=True)
    del df
        
    
    #Converting to numpy array
    dfnew['Reaction Time'] = np.array(dfnew['Reaction Time'], dtype=np.float64)
    
    MeanRT = dfnew['Reaction Time'].mean()
    
    RT_Total.loc[sub] = MeanRT
    
#Reaction Time correlation with Age
RTS = RT_Total['Reaction Time']
slope, intercept, r, p, stderr = scipy.stats.linregress(RTS, age)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}, p={p:.3f}'
fig, ax = plt.subplots()
ax.plot(RTS, age, linewidth=0, marker='s', label='Data points')
ax.plot(RTS, intercept + slope * RTS, label=line)
ax.set_xlabel('Reaction Time')
ax.set_ylabel('Age (years)')
plt.title(label = 'Motor task Reaction Time and age')

legend = ax.legend(frameon = 1)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
ax.grid(color = 'grey')
plt.show()
            

#Reaction Time correlation with MRBD    
print ('MRBD correlation with motor performance')
for chan in range(8):
    print (Topography[chan])
    MRBDS = MRBD_Total[Topography[chan]]
    slope, intercept, r, p, stderr = scipy.stats.linregress(MRBDS, RTS)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}, p={p:.3f}'
    fig, ax = plt.subplots()
    ax.plot(MRBDS, RTS, linewidth=0, marker='s', label='Data points')
    ax.plot(MRBDS, intercept + slope * MRBDS, label=line)
    ax.set_xlabel('MRBD(%)')
    ax.set_ylabel('Reaction Time')
    plt.xlim([-60,42])
    plt.title(label = 'Motor task Reaction Time and MRBD at electrode ' + Topography[chan])
    
    legend = ax.legend(frameon = 1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    ax.grid(color = 'grey')
    plt.show()


#MRBD correlation with PPT

PPT = {
        "AD0109": [9,6,6,21,18],
        "RP0129": [14,13,9,36,26],
        "KM0404": [14,12,10,36,25],
        "CD1107": [16,15,14,45,29],
        "FB0901": [16,13,12,41,29],
        "PB0526": [15,13,11,39,30],
        "MB0522": [14,11,9,34,20],
        "FB0210": [17,14,12,43,30],
        "DF0720": [12,12,11,35,28],
        "PT0728": [12,11,11,34,26],
        "CB0724": [14,14,11,39,32],
        "JM0410": [14,12,9,35,25],
        "GM0804": [10,12,10,32,32],
        "TN0118": [13,14,13,40,34],
        "ME0212": [11,8,8,27,17]     
        }
Measure = ['Right Hand', 'Left Hand', 'Both Hands', 'Right + Left + Both Hands', 'Assembly']
print ('MRBD correlation with PPT')
for measure in range(5):
    Value = []
    print (Measure[measure])
    for sub in SubjectID:
        Value.append(PPT[sub][measure])    
    for chan in range(8):
        print (Topography[chan])
        x = MRBD_Total[Topography[chan]]
        y = Value
        slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
        line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}, p={p:.3f}'
        fig, ax = plt.subplots()
        ax.plot(x, y, linewidth=0, marker='s', label='Data points')
        ax.plot(x, intercept + slope * x, label=line)
        ax.set_xlabel('MRBD(%)')
        ax.set_ylabel('PPT Score (points)')
        plt.xlim([-60,42])
        plt.title(label = 'PPT ' + Measure[measure] + ' and MRBD at electrode ' + Topography[chan])
        
        legend = ax.legend(frameon = 1)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        ax.grid(color = 'grey')
        plt.show()



#Correlate MRBD with Box and Block Test
BBT = {
        "AD0109": [56,50],
        "RP0129": [58,53],
        "KM0404": [54,60],
        "CD1107": [53,59],
        "FB0901": [60,64],
        "PB0526": [65,61],
        "MB0522": [52,43],
        "FB0210": [63,65],
        "DF0720": [46,45],
        "PT0728": [53,62],
        "CB0724": [63,62],
        "JM0410": [51,55],
        "GM0804": [59,62],
        "TN0118": [62,63],
        "ME0212": [51,54]     
        }

Measure = ['Right Hand', 'Left Hand']
print ('MRBD correlation with BBT')
plt.figure(fig, figsize= (15,15))
subp = 331
fig = fig+1
for measure in range(2):
    Value = []
    print (Measure[measure])
    for sub in SubjectID:
        Value.append(BBT[sub][measure])    
    for chan in range(8):
        print (Topography[chan])
        x = MRBD_Total[Topography[chan]]
        y = Value
        slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
        line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}, p={p:.3f}'
        fig, ax = plt.subplots(subp+chan+1)
        ax.plot(x, y, linewidth=0, marker='s', label='Data points')
        ax.plot(x, intercept + slope * x, label=line)
        ax.set_xlabel('MRBD(%)')
        ax.set_ylabel('BBT Score (points)')
        plt.xlim([-60,42])
        plt.title(label = 'BBT ' + Measure[measure] + ' and MRBD at electrodel ' + Topography[chan])
        
        legend = ax.legend(frameon = 1)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        ax.grid(color = 'grey')
        plt.show()


