 # -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 18:16:36 2023

@author: Kenya
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams

import seaborn as sns
import scipy.io
from scipy import stats

plt.style.use('seaborn-colorblind')
#
sns.set_style('whitegrid')

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
AgeOld = [69.7, 76.6, 67.5, 66, 67.1, 66.5, 75.4, 66.8, 66.5, 71.5, 66.6, 65.9, 78.6, 72.2, 69.2]

Young = {
        "AB0420": 20.8,
        "DM0530": 31.9,
        "EC0923": 20.2,
        "ES0501": 21.2,
        "JZ0118": 29.8,
        "KM0123": 24.2,
        "LL0717": 21.5,
        "MA0901": 23.1,
        "MM0929": 26.5,
        "MW0516": 20.5,
        "PV0518": 24.8,
        "SF0429": 20.5,
        "SL1227": 25.8,
        "VN1121": 19.2,
        "YH1204": 26.3     
        }


YoungSub = list(Young)
AgeYoung = list(Young.values())

Channels = ['FC1', 'C3', 'CP5', 'CP1', 'FC3', 'C5', 'C1', 'CP3']
Topography = ['FC3', 'FC1', 'C5', 'C3', 'C1', 'CP5', 'CP3', 'CP1']
time = ['Baseline', 'Post-15min', 'Post-45min']
Sessions = ['S1', 'S2', 'S3']
fig = 0

Young_Total= pd.DataFrame(columns = Channels)
Old_Total = pd.DataFrame(columns = Channels)
MRBD_Young = pd.DataFrame(columns = Channels)
MRBD_Old = pd.DataFrame(columns = Channels)

#Loop for subjects
for s in range(15):
    MeanMRBDsYoung = []
    MeanMRBDsOld = []
    sub = SubjectID[s]
    subyoung = YoungSub[s]
    print(sub)
    ses = 1
    #Loading mat files
    OldMat = scipy.io.loadmat('MeanMRBD_' + sub +'_' + Sessions[ses] + '_Baseline.mat')
    YoungMat = scipy.io.loadmat('MeanMRBD_' + subyoung +'_Baseline.mat')
    
    Old = pd.DataFrame()
    Young = pd.DataFrame()
    
    for chan in range(8):
        print (Channels[chan])
        Old[Channels[chan]] = OldMat['Value'][chan:len(OldMat['Value']):8,1]
        Young[Channels[chan]] = YoungMat['Value'][chan:len(YoungMat['Value']):8,1]
       
        #Remove outliers
        
        #Conidtion that outliers that stand 3 SD from the mean
        conditionold = np.mean(Old[Channels[chan]])+ (np.std(Old[Channels[chan]])*3)
        conditionyoung = np.mean(Young[Channels[chan]])+ (np.std(Young[Channels[chan]])*3)
        
        #Printing which outliers were removed
        for k in range(len(Old)):
            if Old[Channels[chan]][k] > conditionold:
                print('Error Removed outlier: Trial #' + str(k+5) + ' Value: ' + str(Old[Channels[chan]][k]) + ' ' + sub + ' Old')
        
        Old[Channels[chan]] = Old[Channels[chan]].loc[Old[Channels[chan]] < conditionold]
        
        MRBD = Old[Channels[chan]].mean()
        MeanMRBDsOld.append(MRBD)
        
        
        #Printing which outliers were removed
        for k in range(len(Young)):
            if Young[Channels[chan]][k] > conditionyoung:
                print('Error Removed outlier: Trial #' + str(k+5) + ' Value: ' + str(Young[Channels[chan]][k]) + ' ' + subyoung + ' Young' )
        
        Young[Channels[chan]] = Young[Channels[chan]].loc[Young[Channels[chan]] < conditionyoung]
        
        MRBD = Young[Channels[chan]].mean()
        MeanMRBDsYoung.append(MRBD)
    
    MRBD_Old.loc[sub] = MeanMRBDsOld
    MRBD_Young.loc[sub] = MeanMRBDsYoung
        
    
    Young_Total = pd.concat([Young,Young_Total])
    Old_Total = pd.concat([Old,Old_Total])
    
Total = pd.DataFrame (columns = ['Value', 'Time', 'Stimulation', 'Electrode'])


"""
Group-level
"""

plt.figure(fig, figsize= (17,12))
subp = 331
fig = fig+1

maxold = max(Old_Total.max())
maxyoung = max(Young_Total.max())
ymax = max(maxold,maxyoung)

minold = min(Old_Total.min())
minyoung = min(Young_Total.min())
ymin = min(minold,minyoung)


for chan in range(8):
    print (Topography[chan])
    dfold = pd.DataFrame(columns = ['Value', 'Group', 'Electrode'])   
    dfyoung = pd.DataFrame(columns = ['Value', 'Group', 'Electrode'])  
    dfold['Value'] = Old_Total[Topography[chan]]
    dfold['Group']= 'Old'
    dfold['Electrode'] = Topography[chan]
    dfyoung['Value'] = Young_Total[Topography[chan]]
    dfyoung['Group'] = 'Young'
    dfyoung['Electrode'] = Topography[chan]
    dfold = dfold.dropna()
    dfyoung = dfyoung.dropna()
    
    
    dftotal = pd.DataFrame(columns = ['Value', 'Group', 'Electrode'])  
    dftotal = pd.concat([dfold,dfyoung], ignore_index=True)
    Total = pd.concat([Total, dftotal], ignore_index = True)
    

    plt.subplot(subp+chan+1)
    plt.ylim(ymin-10,ymax+70)
    sns.boxplot(data=dftotal, x='Group',y='Value', palette=("Set2"), flierprops = dict(markerfacecolor = '0.50', markersize = 2)).set(title= Topography[chan], xlabel=None, ylabel='MRBD(%)')
    p_ovy = stats.ttest_ind(dfold['Value'],dfyoung['Value'], equal_var= False)[1]
    
    p_values = [p_ovy]
    print('T-test Young vs Old: ' + str(p_ovy))
    # Get the y-axis limits
    y_range = ymax - ymin
    
    levels = [1]
    #Plotting significance bars
    for n in range(1):
        if p_values[n] < 0.05:
            level = levels[n]
            x1 = 0
            x2 = 1-n
            # Plot the bar
            bar_height = (y_range * 0.07 * level) + ymax
            bar_tips = bar_height - (y_range * 0.02)
            plt.plot(
                [x1, x1, x2, x2],
                [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
            )
            # Significance level
            p = p_values[n]
            if p < 0.001:
                sig_symbol = '***'
            elif p < 0.01:
                sig_symbol = '**'
            elif p < 0.05:
                sig_symbol = '*'
            text_height = bar_height + (y_range * 0.01)
            plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')
     

#Correlation MRBD & age


AllMRBDs = pd.DataFrame(columns = Channels)
AllMRBDs = pd.concat([MRBD_Young, MRBD_Old], ignore_index=True)

AllAge = np.array(AgeYoung + AgeOld)

print ('MRBD correlation with ')
for chan in range(8):
    print (Topography[chan])
    x = AllAge
    y = AllMRBDs[Topography[chan]]
    slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}, p={p:.2f}'
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=0, marker='x', label='Data points')
    ax.plot(x, intercept + slope * x, label=line, linewidth=3)
    ax.set_xlabel('Age')
    ax.set_ylabel('MRBD(%)')
    plt.title(label = 'Age correlation with MRBD Channel ' + Topography[chan])
    
    legend = ax.legend(frameon = 1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    ax.grid(color = 'grey')
    plt.show()

import pylab

scipy.stats.probplot(AllAge, dist='norm', plot=pylab)
pylab.show()

ax = sns.histplot(AllMRBDs)













