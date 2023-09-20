# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:46:38 2023

@author: Kenya
"""

import numpy as np
import pandas as pd

import scipy.io
from statsmodels.stats.anova import AnovaRM

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

Channels = ['FC1', 'C3', 'CP5', 'CP1', 'FC3', 'C5', 'C1', 'CP3']
Topography = ['FC3', 'FC1', 'C5', 'C3', 'C1', 'CP5', 'CP3', 'CP1']
time = ['Baseline', 'Post-15min', 'Post-45min']
Sessions = ['S1', 'S2', 'S3']
fig = 0

TotalMRBD = pd.DataFrame (columns = ['Value', 'Time', 'Subject', 'Stimulation', 'Electrode'])

##############################################################################
#                       MRBD
##############################################################################

#Loop for subjects
for s in range(15):
    sub = SubjectID[s]
    #Loop for sessions
    for ses in range(3):
        #Loading mat files
        BaselineMat = scipy.io.loadmat('MeanMRBD_' + sub +'_' + Sessions[ses] + '_Baseline.mat')
        FifteenMat = scipy.io.loadmat('MeanMRBD_' + sub +'_' + Sessions[ses] + '_15min.mat')
        FortyfiveMat = scipy.io.loadmat('MeanMRBD_' + sub +'_' + Sessions[ses] + '_45min.mat')
        
        Baseline = pd.DataFrame()
        Fifteen = pd.DataFrame()
        Fortyfive = pd.DataFrame()
        
        for chan in range(8):
            Baseline[Channels[chan]] = BaselineMat['Value'][chan:len(BaselineMat['Value']):8,1]
            Fifteen[Channels[chan]] = FifteenMat['Value'][chan:len(FifteenMat['Value']):8,1]
            Fortyfive[Channels[chan]] = FortyfiveMat['Value'][chan:len(FortyfiveMat['Value']):8,1]
           
            #Remove outliers
            
            #Conidtion that outliers that stand 3 SD from the mean
            conditionbase = np.mean(Baseline[Channels[chan]])+ (np.std(Baseline[Channels[chan]])*3)
            condition15 = np.mean(Fifteen[Channels[chan]])+ (np.std(Fifteen[Channels[chan]])*3)
            condition45 = np.mean(Fortyfive[Channels[chan]])+ (np.std(Fortyfive[Channels[chan]])*3)
            
            Baseline[Channels[chan]] = Baseline[Channels[chan]].loc[Baseline[Channels[chan]] < conditionbase]
            Fifteen[Channels[chan]] = Fifteen[Channels[chan]].loc[Fifteen[Channels[chan]] < condition15]
            Fortyfive[Channels[chan]] = Fortyfive[Channels[chan]].loc[Fortyfive[Channels[chan]] < condition45]

        dfbase = pd.DataFrame(columns = ['Value', 'Time', 'Subject'])   
        dffifteen = pd.DataFrame(columns = ['Value', 'Time', 'Subject'])  
        dffortyfive = pd.DataFrame(columns = ['Value', 'Time', 'Subject'])
        
        for chan in range(8):
            dfbase['Value'] = Baseline[Topography[chan]]
            dfbase['Electrode'] = Topography[chan]
            dfbase['Time']= time[0]
            dfbase['Subject'] = sub
            dfbase['Stimulation'] = Dict[sub][ses]
            dffifteen['Value'] = Fifteen[Topography[chan]]
            dffifteen['Electrode'] = Topography[chan]
            dffifteen['Time'] = time[1]
            dffifteen['Subject'] = sub
            dffifteen['Stimulation'] = Dict[sub][ses]
            dffortyfive['Value']= Fortyfive[Topography[chan]]
            dffortyfive['Electrode'] = Topography[chan]
            dffortyfive['Time'] = time[2]
            dffortyfive['Subject'] = sub
            dffortyfive['Stimulation'] = Dict[sub][ses]
            dfbase = dfbase.dropna()
            dffifteen = dffifteen.dropna()
            dffortyfive = dffortyfive.dropna()
            
            
            dftotal = pd.DataFrame(columns = ['Value', 'Electrode', 'Time', 'Subject'])  
            dftotal = pd.concat([dfbase,dffifteen,dffortyfive], ignore_index=True)
            TotalMRBD = pd.concat([TotalMRBD, dftotal], ignore_index=True)

print('MRBD')
for chan in range(8):
    Electrode = TotalMRBD[np.isin(TotalMRBD, [Topography[chan]]).any(axis=1)]
    print (Topography[chan])
    anovares = AnovaRM(Electrode, 'Value', 'Subject', within = ['Time', 'Stimulation'], aggregate_func = 'mean').fit()
    print(anovares)










