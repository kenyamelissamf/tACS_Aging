# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:26:42 2023

@author: Kenya Morales
@contact: kenyamelissamf@gmail.com
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import scipy.io
from scipy import stats
import statsmodels.stats.multitest
from numpy import std, mean, sqrt


def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)


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

Channels = ['FC1', 'C3', 'CP5', 'CP1', 'FC3', 'C5', 'C1', 'CP3']
Topography = ['FC3', 'FC1', 'C5', 'C3', 'C1', 'CP5', 'CP3', 'CP1']
time = ['Baseline', 'Post-15min', 'Post-45min']
Sessions = ['S1', 'S2', 'S3']
fig = 0

Baseline_20= pd.DataFrame(columns = Channels)
Fifteen_20 =pd.DataFrame(columns = Channels)
Fortyfive_20 = pd.DataFrame(columns = Channels)
Baseline_70= pd.DataFrame(columns = Channels)
Fifteen_70 =pd.DataFrame(columns = Channels)
Fortyfive_70 = pd.DataFrame(columns = Channels)
Baseline_Sham= pd.DataFrame(columns = Channels)
Fifteen_Sham =pd.DataFrame(columns = Channels)
Fortyfive_Sham = pd.DataFrame(columns = Channels)



#Loop for subjects
for s in range(15):
    sub = SubjectID[s]
    #Loop for sessions
    for ses in range(3):
        print(sub + ' ' + Dict[sub][ses])
        #Loading mat files
        BaselineMat = scipy.io.loadmat('BetaPower_' + sub +'_' + Sessions[ses] + '_Baseline.mat')
        FifteenMat = scipy.io.loadmat('BetaPower_' + sub +'_' + Sessions[ses] + '_15min.mat')
        FortyfiveMat = scipy.io.loadmat('BetaPower_' + sub +'_' + Sessions[ses] + '_45min.mat')
        
        Baseline = pd.DataFrame()
        Fifteen = pd.DataFrame()
        Fortyfive = pd.DataFrame()
        
        for chan in range(8):
            print (Channels[chan])
            Baseline[Channels[chan]] = BaselineMat['TF'][chan:len(BaselineMat['TF']):8,1]
            Fifteen[Channels[chan]] = FifteenMat['TF'][chan:len(FifteenMat['TF']):8,1]
            Fortyfive[Channels[chan]] = FortyfiveMat['TF'][chan:len(FortyfiveMat['TF']):8,1]
           
            #Remove outliers
            
            #Conidtion that outliers that stand 3 SD from the mean
            conditionbase = np.mean(Baseline[Channels[chan]])+ (np.std(Baseline[Channels[chan]])*3)
            condition15 = np.mean(Fifteen[Channels[chan]])+ (np.std(Fifteen[Channels[chan]])*3)
            condition45 = np.mean(Fortyfive[Channels[chan]])+ (np.std(Fortyfive[Channels[chan]])*3)
            
            #Printing which outliers were removed
            for k in range(len(Baseline)):
                if Baseline[Channels[chan]][k] > conditionbase:
                    print('Error Removed outlier: Trial #' + str(k+5) + ' Value: ' + str(Baseline[Channels[chan]][k]) + ' ' + sub + ' ' + Dict[sub][ses] + ' Baseline' )
            
            Baseline[Channels[chan]] = Baseline[Channels[chan]].loc[Baseline[Channels[chan]] < conditionbase]
            
            
            #Printing which outliers were removed
            for k in range(len(Fifteen)):
                if Fifteen[Channels[chan]][k] > condition15:
                    print('Error Removed outlier: Trial #' + str(k+5) + ' Value: ' + str(Fifteen[Channels[chan]][k]) + ' ' + sub + ' ' + Dict[sub][ses] + ' 15min' )
            
            Fifteen[Channels[chan]] = Fifteen[Channels[chan]].loc[Fifteen[Channels[chan]] < condition15]

            #Printing which outliers were removed
            for k in range(len(Fortyfive)):
                if Fortyfive[Channels[chan]][k] > condition45:
                    print('Error Removed outlier: Trial #' + str(k+5) + ' Value: ' + str(Fortyfive[Channels[chan]][k]) + ' ' + sub + ' ' + Dict[sub][ses] + ' 45min' )
            
            Fortyfive[Channels[chan]] = Fortyfive[Channels[chan]].loc[Fortyfive[Channels[chan]] < condition45]

        maxbase = max(Baseline.max())
        max15 = max(Fifteen.max())
        max45 = max(Fortyfive.max())
        ymax = max(maxbase,max15,max45)
        
        minbase = min(Baseline.min())
        min15 = min(Fifteen.min())
        min45 = min(Fortyfive.min())
        ymin = min(minbase,min15,min45)
        
        if Dict[sub][ses] == '20 Hz':
            Baseline_20 = pd.concat([Baseline,Baseline_20])
            Fifteen_20 = pd.concat([Fifteen,Fifteen_20])
            Fortyfive_20 = pd.concat([Fortyfive,Fortyfive_20])
        elif Dict[sub][ses] == '70 Hz':
            Baseline_70 = pd.concat([Baseline,Baseline_70])
            Fifteen_70 = pd.concat([Fifteen,Fifteen_70])
            Fortyfive_70 = pd.concat([Fortyfive,Fortyfive_70])
        elif Dict[sub][ses] == 'Sham':
            if sub == 'TN0118':
                Baseline_Sham = Baseline_Sham
            else:
                Baseline_Sham = pd.concat([Baseline,Baseline_Sham])
                Fifteen_Sham = pd.concat([Fifteen,Fifteen_Sham])
                Fortyfive_Sham = pd.concat([Fortyfive,Fortyfive_Sham])
        
        dfbase = pd.DataFrame(columns = ['Beta Power', 'Time'])   
        dffifteen = pd.DataFrame(columns = ['Beta Power', 'Time'])  
        dffortyfive = pd.DataFrame(columns = ['Beta Power', 'Time'])
        
        
        plt.figure(fig, figsize= (20,15))
        subp = 331
        fig = fig+1
        
        for chan in range(8):
            print (Topography[chan])
            dfbase['Beta Power'] = Baseline[Topography[chan]]
            dfbase['Time']= time[0]
            dffifteen['Beta Power'] = Fifteen[Topography[chan]]
            dffifteen['Time'] = time[1]
            dffortyfive['Beta Power']= Fortyfive[Topography[chan]]
            dffortyfive['Time'] = time[2]
            dfbase = dfbase.dropna()
            dffifteen = dffifteen.dropna()
            dffortyfive = dffortyfive.dropna()
            
            
            dftotal = pd.DataFrame(columns = ['Beta Power', 'Time'])  
            dftotal = pd.concat([dfbase,dffifteen,dffortyfive], ignore_index=True)
        
            plt.subplot(subp+chan+1)
            plt.ylim(ymin-(ymin*0.50),ymax+(ymax*0.20))
            sns.boxplot(data=dftotal, x='Time',y='Beta Power', palette=("viridis"), flierprops = dict(markerfacecolor = '0.50', markersize = 2)).set(title= sub + ' MRBD(%) ' + Topography[chan] + ' ' + Dict[sub][ses])
            
            
            p_fifteen = stats.ttest_ind(dfbase['Beta Power'],dffifteen['Beta Power'])[1]
            p_fortyfive = stats.ttest_ind(dfbase['Beta Power'],dffortyfive['Beta Power'])[1]
            
            p_values = [p_fortyfive, p_fifteen]
            decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=p_values, alpha=0.05, method='bonferroni')
            [p_fortyfive, p_fifteen] = adj_pvals
            p_values = adj_pvals
            print('T-test Error Base vs Fifteen: ' + str(p_fifteen))
            print('T-test Error Base vs Fortyfive: ' + str(p_fortyfive))
            
            # Get the y-axis limits
            y_range = ymax - ymin
            
            levels = [2,1]
            #Plotting significance bars
            for n in range(2):
                if p_values[n] < 0.05:
                    level = levels[n]
                    x1 = 0
                    x2 = 2-n
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
                
        #del Fifteen, Fortyfive, Baseline, BaselineMat, FifteenMat, FortyfiveMat

Total_20 = pd.DataFrame (columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode','Normalized'])
Total_70 = pd.DataFrame (columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode','Normalized'])
Total_Sham = pd.DataFrame (columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode','Normalized'])

"""
20 Hz Group-level
"""

print('20 Hz Group-level')

plt.figure(fig, figsize= (20,15))
subp = 331
fig = fig+1

maxbase = max(Baseline_20.max())
max15 = max(Fifteen_20.max())
max45 = max(Fortyfive_20.max())
ymax = max(maxbase,max15,max45)

minbase = min(Baseline_20.min())
min15 = min(Fifteen_20.min())
min45 = min(Fortyfive_20.min())
ymin = min(minbase,min15,min45)

# Plot group-level 20 Hz
for chan in range(8):
    print (Topography[chan])
    del dfbase, dffifteen, dffortyfive
    dfbase = pd.DataFrame(columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode'])   
    dffifteen = pd.DataFrame(columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode'])  
    dffortyfive = pd.DataFrame(columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode']) 
    dfbase['Beta Power'] = Baseline_20[Topography[chan]]
    dfbase['Time']= time[0]
    dfbase['Stimulation'] = '20 Hz'
    dfbase['Electrode'] = Topography[chan]
    dffifteen['Beta Power'] = Fifteen_20[Topography[chan]]
    dffifteen['Time'] = time[1]
    dffifteen['Stimulation'] = '20 Hz'
    dffifteen['Electrode'] = Topography[chan]
    dffortyfive['Beta Power']= Fortyfive_20[Topography[chan]]
    dffortyfive['Time'] = time[2]
    dffortyfive['Stimulation'] = '20 Hz'
    dffortyfive['Electrode'] = Topography[chan]
    dfbase = dfbase.dropna()
    dffifteen = dffifteen.dropna()
    dffortyfive = dffortyfive.dropna()
    
    
    dftotal = pd.DataFrame(columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode'])  
    dftotal = pd.concat([dfbase,dffifteen,dffortyfive], ignore_index=True)
    Total_20 = pd.concat([Total_20, dftotal], ignore_index = True)
    

    plt.subplot(subp+chan+1)
    plt.ylim(ymin-(ymin*0.50),ymax+(ymax*0.20))
    sns.boxplot(data=dftotal, x='Time',y='Beta Power', palette=("rocket"), flierprops = dict(markerfacecolor = '0.50', markersize = 2)).set(title= ' Group-level MRBD(%) ' + Topography[chan] + ' 20 Hz')
    
    
    p_fifteen = stats.ttest_ind(dfbase['Beta Power'],dffifteen['Beta Power'])[1]
    p_fortyfive = stats.ttest_ind(dfbase['Beta Power'],dffortyfive['Beta Power'])[1]
    
    p_values = [p_fortyfive, p_fifteen]
    decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=p_values, alpha=0.05, method='bonferroni')
    [p_fortyfive, p_fifteen] = adj_pvals
    print('T-test Error Base vs Fifteen: ' + str(p_fifteen))
    print('T-test Error Base vs Fortyfive: ' + str(p_fortyfive))
    # Get the y-axis limits
    y_range = ymax - ymin
    
    levels = [2,1]
    #Plotting significance bars
    for n in range(2):
        if p_values[n] < 0.05:
            level = levels[n]
            x1 = 0
            x2 = 2-n
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
     
        
"""
70 Hz Group-level
"""
    
print('70 Hz Group-level')

plt.figure(fig, figsize= (20,15))
subp = 331
fig = fig+1

maxbase = max(Baseline_70.max())
max15 = max(Fifteen_70.max())
max45 = max(Fortyfive_70.max())
ymax = max(maxbase,max15,max45)

minbase = min(Baseline_70.min())
min15 = min(Fifteen_70.min())
min45 = min(Fortyfive_70.min())
ymin = min(minbase,min15,min45)

# Plot group-level 70 Hz
for chan in range(8):
    print (Topography[chan])
    del dfbase, dffifteen, dffortyfive
    dfbase = pd.DataFrame(columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode'])   
    dffifteen = pd.DataFrame(columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode'])  
    dffortyfive = pd.DataFrame(columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode']) 
    dfbase['Beta Power'] = Baseline_70[Topography[chan]]
    dfbase['Time']= time[0]
    dfbase['Stimulation'] = '70 Hz'
    dfbase['Electrode'] = Topography[chan]
    dffifteen['Beta Power'] = Fifteen_70[Topography[chan]]
    dffifteen['Time'] = time[1]
    dffifteen['Stimulation'] = '70 Hz'
    dffifteen['Electrode'] = Topography[chan]
    dffortyfive['Beta Power']= Fortyfive_70[Topography[chan]]
    dffortyfive['Time'] = time[2]
    dffortyfive['Stimulation'] = '70 Hz'
    dffortyfive['Electrode'] = Topography[chan]
    
    dfbase = dfbase.dropna()
    dffifteen = dffifteen.dropna()
    dffortyfive = dffortyfive.dropna()
    
    dftotal = pd.DataFrame(columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode'])  
    dftotal = pd.concat([dfbase,dffifteen,dffortyfive], ignore_index=True)
    Total_70 = pd.concat([Total_70, dftotal], ignore_index = True)

    plt.subplot(subp+chan+1)
    plt.ylim(ymin-(ymin*0.50),ymax+(ymax*0.20))
    sns.boxplot(data=dftotal, x='Time',y='Beta Power', palette=("rocket"), flierprops = dict(markerfacecolor = '0.50', markersize = 2)).set(title= ' Group-level MRBD(%) ' + Topography[chan] + ' 70 Hz')
    
    
    p_fifteen = stats.ttest_ind(dfbase['Beta Power'],dffifteen['Beta Power'])[1]
    p_fortyfive = stats.ttest_ind(dfbase['Beta Power'],dffortyfive['Beta Power'])[1]
    
    p_values = [p_fortyfive, p_fifteen]
    decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=p_values, alpha=0.05, method='bonferroni')
    [p_fortyfive, p_fifteen] = adj_pvals
    p_values = adj_pvals
    print('T-test Error Base vs Fifteen: ' + str(p_fifteen))
    print('T-test Error Base vs Fortyfive: ' + str(p_fortyfive))
    # Get the y-axis limits
    y_range = ymax - ymin
    
    levels = [2,1]
    #Plotting significance bars
    for n in range(2):
        if p_values[n] < 0.05:
            level = levels[n]
            x1 = 0
            x2 = 2-n
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
   

"""
Sham Group-level
"""
    
print('Sham Group-level')

plt.figure(fig, figsize= (20,15))
subp = 331
fig = fig+1

maxbase = max(Baseline_Sham.max())
max15 = max(Fifteen_Sham.max())
max45 = max(Fortyfive_Sham.max())
ymax = max(maxbase,max15,max45)

minbase = min(Baseline_Sham.min())
min15 = min(Fifteen_Sham.min())
min45 = min(Fortyfive_Sham.min())
ymin = min(minbase,min15,min45)

# Plot group-level 70 Hz
for chan in range(8):
    print (Topography[chan])
    del dfbase, dffifteen, dffortyfive
    dfbase = pd.DataFrame(columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode'])   
    dffifteen = pd.DataFrame(columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode'])  
    dffortyfive = pd.DataFrame(columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode']) 
    dfbase['Beta Power'] = Baseline_Sham[Topography[chan]]
    dfbase['Time']= time[0]
    dfbase['Stimulation'] = 'Sham'
    dfbase['Electrode'] = Topography[chan]
    dffifteen['Beta Power'] = Fifteen_Sham[Topography[chan]]
    dffifteen['Time'] = time[1]
    dffifteen['Stimulation'] = 'Sham'
    dffifteen['Electrode'] = Topography[chan]
    dffortyfive['Beta Power']= Fortyfive_Sham[Topography[chan]]
    dffortyfive['Time'] = time[2]
    dffortyfive['Stimulation'] = 'Sham'
    dffortyfive['Electrode'] = Topography[chan]
    
    dfbase = dfbase.dropna()
    dffifteen = dffifteen.dropna()
    dffortyfive = dffortyfive.dropna()
    
    dftotal = pd.DataFrame(columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode'])  
    dftotal = pd.concat([dfbase,dffifteen,dffortyfive], ignore_index=True)
    Total_Sham = pd.concat([Total_Sham, dftotal], ignore_index = True)

    plt.subplot(subp+chan+1)
    plt.ylim(ymin-(ymin*0.50),ymax+(ymax*0.20))
    sns.boxplot(data=dftotal, x='Time',y='Beta Power', palette=("rocket"), flierprops = dict(markerfacecolor = '0.50', markersize = 2)).set(title= ' Group-level MRBD(%) ' + Topography[chan] + ' Sham')
    
    
    p_fifteen = stats.ttest_ind(dfbase['Beta Power'],dffifteen['Beta Power'])[1]
    p_fortyfive = stats.ttest_ind(dfbase['Beta Power'],dffortyfive['Beta Power'])[1]
    
    p_values = [p_fortyfive, p_fifteen]
    decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=p_values, alpha=0.05, method='bonferroni')
    [p_fortyfive, p_fifteen] = adj_pvals
    p_values = adj_pvals
    print('T-test Error Base vs Fifteen: ' + str(p_fifteen))
    print('T-test Error Base vs Fortyfive: ' + str(p_fortyfive))
    # Get the y-axis limits
    y_range = ymax - ymin
    
    levels = [2,1]
    #Plotting significance bar
    for n in range(2):
        if p_values[n] < 0.05:
            level = levels[n]
            x1 = 0
            x2 = 2-n
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


#Cat-plot
ymin=-30
ymax=20

Total = pd.DataFrame (columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode'])
Total = pd.concat([Total_20, Total_70, Total_Sham], ignore_index = True)

plt.figure(fig)
fig = fig+1
#plt.ylim(ymin,ymax)
First = Total[np.isin(Total, ['FC3','FC1']).any(axis=1)]

sns.set_theme(style="whitegrid")
sns.catplot(
    data = First, x = 'Time', y='Beta Power',
    hue='Stimulation', capsize=.05, palette='magma', col = 'Electrode',
    kind = 'point', errorbar = 'sd'
    )


plt.figure(fig)
fig = fig+1
#plt.ylim(ymin,ymax)
Second = Total[np.isin(Total, ['C5','C3', 'C1']).any(axis=1)]

sns.set_theme(style="whitegrid")
sns.catplot(
    data = Second, x = 'Time', y='Beta Power',
    hue='Stimulation', capsize=.05, palette='magma', col = 'Electrode',
    kind = 'point', errorbar = 'sd'
    )


plt.figure(fig)
fig = fig+1
#plt.ylim(ymin,ymax)
Third = Total[np.isin(Total, ['CP5','CP3', 'CP1']).any(axis=1)]

sns.set_theme(style="whitegrid")
sns.catplot(
    data = Third, x = 'Time', y='Beta Power',
    hue='Stimulation', capsize=.05, palette='magma', col = 'Electrode',
    kind = 'point', errorbar = 'sd'
    )
#plt.ylim(ymin,ymax)

#Normalize data

Stim = ['20 Hz', '70 Hz', 'Sham']

grouped_electrodes = Total.groupby('Electrode')
Normalized = pd.DataFrame (columns = ['Beta Power', 'Time', 'Stimulation', 'Electrode','Normalized'])
for chan in range(8):
    Electrode = grouped_electrodes.get_group(Topography[chan])
    grouped_session = Electrode.groupby('Stimulation')
    for ses in range(3):
        Session = grouped_session.get_group(Stim[ses])
        mean = Session.loc[Session['Time']== 'Baseline', 'Beta Power'].mean()
        Session['Normalized'] = Session['Beta Power']-(mean)
    
        Normalized = pd.concat([Normalized, Session], ignore_index = True)
    
ymin=-5e-14
ymax=6e-14

plt.figure(fig)
fig = fig+1
plt.ylim(ymin,ymax)
First = Normalized[np.isin(Normalized, ['FC3','FC1']).any(axis=1)]

sns.set_theme(style="whitegrid")
ax = sns.catplot(
    data = First, x = 'Time', y='Normalized',
    hue='Stimulation', capsize=.05, palette='magma', col = 'Electrode',
    kind = 'point', errorbar = 'se', aspect=0.65
    )
ax.set(xlabel='Time', ylabel='Beta Power\n Normalized to baseline')

plt.figure(fig, figsize= (15,15))
plt.ylim(ymin,ymax)
fig = fig+1
Second = Normalized[np.isin(Normalized, ['C5','C3', 'C1']).any(axis=1)]

sns.set_theme(style="whitegrid")
ax = sns.catplot(
    data = Second, x = 'Time', y='Normalized',
    hue='Stimulation', capsize=.05, palette='magma', col = 'Electrode',
    kind = 'point', errorbar = 'se',aspect=0.65
    )
ax.set(xlabel='Time', ylabel='Beta Power\n Normalized to baseline')


plt.figure(fig, figsize= (15,15))
plt.ylim(ymin,ymax)
fig = fig+1
Third = Normalized[np.isin(Normalized, ['CP5','CP3', 'CP1']).any(axis=1)]

sns.set_theme(style="whitegrid")
ax = sns.catplot(
    data = Third, x = 'Time', y='Normalized',
    hue='Stimulation', capsize=.05, palette='magma', col = 'Electrode',
    kind = 'point', errorbar = 'se',aspect=0.65
    )
ax.set(xlabel='Time', ylabel='Beta Power\n Normalized to baseline')

plt.ylim(ymin,ymax)

