# -- coding: utf-8 --
"""
Created on Thu May 18 15:13:03 2023

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
from statsmodels.stats.anova import AnovaRM

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
        BaselineMat = scipy.io.loadmat('MeanMRBD_' + sub +'_' + Sessions[ses] + '_Baseline.mat')
        FifteenMat = scipy.io.loadmat('MeanMRBD_' + sub +'_' + Sessions[ses] + '_15min.mat')
        FortyfiveMat = scipy.io.loadmat('MeanMRBD_' + sub +'_' + Sessions[ses] + '_45min.mat')
        
        Baseline = pd.DataFrame()
        Fifteen = pd.DataFrame()
        Fortyfive = pd.DataFrame()
        
        for chan in range(8):
            print (Channels[chan])
            Baseline[Channels[chan]] = BaselineMat['Value'][chan:len(BaselineMat['Value']):8,1]
            Fifteen[Channels[chan]] = FifteenMat['Value'][chan:len(FifteenMat['Value']):8,1]
            Fortyfive[Channels[chan]] = FortyfiveMat['Value'][chan:len(FortyfiveMat['Value']):8,1]
           
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
            if sub == 'AD0109' or sub =='FB0901':
                Baseline_20 = Baseline_20
            else:
                Baseline_20 = pd.concat([Baseline,Baseline_20])
                Fifteen_20 = pd.concat([Fifteen,Fifteen_20])
                Fortyfive_20 = pd.concat([Fortyfive,Fortyfive_20])
        elif Dict[sub][ses] == '70 Hz':
            Baseline_70 = pd.concat([Baseline,Baseline_70])
            Fifteen_70 = pd.concat([Fifteen,Fifteen_70])
            Fortyfive_70 = pd.concat([Fortyfive,Fortyfive_70])
        elif Dict[sub][ses] == 'Sham':
            Baseline_Sham = pd.concat([Baseline,Baseline_Sham])
            Fifteen_Sham = pd.concat([Fifteen,Fifteen_Sham])
            Fortyfive_Sham = pd.concat([Fortyfive,Fortyfive_Sham])
        
        dfbase = pd.DataFrame(columns = ['Value', 'Time', 'Subject'])   
        dffifteen = pd.DataFrame(columns = ['Value', 'Time', 'Subject'])  
        dffortyfive = pd.DataFrame(columns = ['Value', 'Time', 'Subject'])
        
        
        plt.figure(fig, figsize= (20,15))
        subp = 331
        fig = fig+1
        
        for chan in range(8):
            print (Topography[chan])
            dfbase['Value'] = Baseline[Topography[chan]]
            dfbase['Time']= time[0]
            dfbase['Subject'] = sub
            dffifteen['Value'] = Fifteen[Topography[chan]]
            dffifteen['Time'] = time[1]
            dffifteen['Subject'] = sub
            dffortyfive['Value']= Fortyfive[Topography[chan]]
            dffortyfive['Time'] = time[2]
            dffortyfive['Subject'] = sub
            dfbase = dfbase.dropna()
            dffifteen = dffifteen.dropna()
            dffortyfive = dffortyfive.dropna()
            
            
            dftotal = pd.DataFrame(columns = ['Value', 'Time', 'Subject'])  
            dftotal = pd.concat([dfbase,dffifteen,dffortyfive], ignore_index=True)
        
            plt.subplot(subp+chan+1)
            plt.ylim(ymin-5,ymax+70)
            sns.boxplot(data=dftotal, x='Time',y='Value', palette=("viridis"),flierprops = dict(markerfacecolor = '0.50', markersize = 2)).set(title= sub + ' MRBD(%) ' + Topography[chan] + ' ' + Dict[sub][ses])
            
            
            p_fifteen = stats.ttest_ind(dfbase['Value'],dffifteen['Value'])[1]
            p_fortyfive = stats.ttest_ind(dfbase['Value'],dffortyfive['Value'])[1]
            
            p_values = [p_fortyfive, p_fifteen]
            decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=p_values, alpha=0.05, method='bonferroni')
            p_values = adj_pvals
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
                
        #del Fifteen, Fortyfive, Baseline, BaselineMat, FifteenMat, FortyfiveMat

Total_20 = pd.DataFrame (columns = ['Value', 'Time', 'Subject', 'Stimulation', 'Electrode','Normalized'])
Total_70 = pd.DataFrame (columns = ['Value', 'Time', 'Subject', 'Stimulation', 'Electrode','Normalized'])
Total_Sham = pd.DataFrame (columns = ['Value', 'Time', 'Subject', 'Stimulation', 'Electrode','Normalized'])

"""
20 Hz Group-level
"""

print('20 Hz Group-level')

plt.figure(fig, figsize= (15,15))
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
    dfbase = pd.DataFrame(columns = ['Value', 'Time', 'Stimulation', 'Electrode'])   
    dffifteen = pd.DataFrame(columns = ['Value', 'Time', 'Stimulation', 'Electrode'])  
    dffortyfive = pd.DataFrame(columns = ['Value', 'Time', 'Stimulation', 'Electrode']) 
    dfbase['Value'] = Baseline_20[Topography[chan]]
    dfbase['Time']= time[0]
    dfbase['Stimulation'] = '20 Hz'
    dfbase['Electrode'] = Topography[chan]
    dffifteen['Value'] = Fifteen_20[Topography[chan]]
    dffifteen['Time'] = time[1]
    dffifteen['Stimulation'] = '20 Hz'
    dffifteen['Electrode'] = Topography[chan]
    dffortyfive['Value']= Fortyfive_20[Topography[chan]]
    dffortyfive['Time'] = time[2]
    dffortyfive['Stimulation'] = '20 Hz'
    dffortyfive['Electrode'] = Topography[chan]
    dfbase = dfbase.dropna()
    dffifteen = dffifteen.dropna()
    dffortyfive = dffortyfive.dropna()
    
    
    dftotal = pd.DataFrame(columns = ['Value', 'Time', 'Stimulation', 'Electrode'])  
    dftotal = pd.concat([dfbase,dffifteen,dffortyfive], ignore_index=True)
    Total_20 = pd.concat([Total_20, dftotal], ignore_index = True)
    

    plt.subplot(subp+chan+1)
    plt.ylim(ymin-10,ymax+50)
    sns.boxplot(data=dftotal, x='Time',y='Value', palette=("Set2"),flierprops = dict(markerfacecolor = '0.50', markersize = 2)).set(title= Topography[chan] )
     
    p_fifteen = stats.ttest_ind(dfbase['Value'],dffifteen['Value'])[1]
    p_fortyfive = stats.ttest_ind(dfbase['Value'],dffortyfive['Value'])[1]
    
    p_values = [p_fortyfive, p_fifteen]
    decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=p_values, alpha=0.05, method='bonferroni')
    p_values = adj_pvals
    [p_fortyfive, p_fifteen] = adj_pvals
    print('T-test Base vs Fifteen: ' + str(p_fifteen))
    print('T-test Base vs Fortyfive: ' + str(p_fortyfive))
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
            bar_height = (y_range * 0.07 * level) + (ymax-30)
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

plt.figure(fig, figsize= (15,15))
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
    dfbase = pd.DataFrame(columns = ['Value', 'Time', 'Stimulation', 'Electrode'])   
    dffifteen = pd.DataFrame(columns = ['Value', 'Time', 'Stimulation', 'Electrode'])  
    dffortyfive = pd.DataFrame(columns = ['Value', 'Time', 'Stimulation', 'Electrode']) 
    dfbase['Value'] = Baseline_70[Topography[chan]]
    dfbase['Time']= time[0]
    dfbase['Stimulation'] = '70 Hz'
    dfbase['Electrode'] = Topography[chan]
    dffifteen['Value'] = Fifteen_70[Topography[chan]]
    dffifteen['Time'] = time[1]
    dffifteen['Stimulation'] = '70 Hz'
    dffifteen['Electrode'] = Topography[chan]
    dffortyfive['Value']= Fortyfive_70[Topography[chan]]
    dffortyfive['Time'] = time[2]
    dffortyfive['Stimulation'] = '70 Hz'
    dffortyfive['Electrode'] = Topography[chan]
    
    dfbase = dfbase.dropna()
    dffifteen = dffifteen.dropna()
    dffortyfive = dffortyfive.dropna()
    
    dftotal = pd.DataFrame(columns = ['Value', 'Time', 'Stimulation', 'Electrode'])  
    dftotal = pd.concat([dfbase,dffifteen,dffortyfive], ignore_index=True)
    Total_70 = pd.concat([Total_70, dftotal], ignore_index = True)

    plt.subplot(subp+chan+1)
    plt.ylim(ymin-20,ymax+20)
    sns.boxplot(data=dftotal, x='Time',y='Value', palette=("Set2"),flierprops = dict(markerfacecolor = '0.50', markersize = 2)).set(title= Topography[chan])
    
    
    p_fifteen = stats.ttest_ind(dfbase['Value'],dffifteen['Value'])[1]
    p_fortyfive = stats.ttest_ind(dfbase['Value'],dffortyfive['Value'])[1]
    
    p_values = [p_fortyfive, p_fifteen]
    decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=p_values, alpha=0.05, method='bonferroni')
    p_values = adj_pvals
    [p_fortyfive, p_fifteen] = adj_pvals
    print('T-test Base vs Fifteen: ' + str(p_fifteen))
    print('T-test Base vs Fortyfive: ' + str(p_fortyfive))
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
            bar_height = (y_range * 0.07 * level) + (ymax-60)
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

plt.figure(fig, figsize= (15,15))
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
    dfbase = pd.DataFrame(columns = ['Value', 'Time', 'Stimulation', 'Electrode'])   
    dffifteen = pd.DataFrame(columns = ['Value', 'Time', 'Stimulation', 'Electrode'])  
    dffortyfive = pd.DataFrame(columns = ['Value', 'Time', 'Stimulation', 'Electrode']) 
    dfbase['Value'] = Baseline_Sham[Topography[chan]]
    dfbase['Time']= time[0]
    dfbase['Stimulation'] = 'Sham'
    dfbase['Electrode'] = Topography[chan]
    dffifteen['Value'] = Fifteen_Sham[Topography[chan]]
    dffifteen['Time'] = time[1]
    dffifteen['Stimulation'] = 'Sham'
    dffifteen['Electrode'] = Topography[chan]
    dffortyfive['Value']= Fortyfive_Sham[Topography[chan]]
    dffortyfive['Time'] = time[2]
    dffortyfive['Stimulation'] = 'Sham'
    dffortyfive['Electrode'] = Topography[chan]
    
    dfbase = dfbase.dropna()
    dffifteen = dffifteen.dropna()
    dffortyfive = dffortyfive.dropna()
    
    dftotal = pd.DataFrame(columns = ['Value', 'Time', 'Stimulation', 'Electrode'])  
    dftotal = pd.concat([dfbase,dffifteen,dffortyfive], ignore_index=True)
    Total_Sham = pd.concat([Total_Sham, dftotal], ignore_index = True)

    plt.subplot(subp+chan+1)
    plt.ylim(ymin-15,ymax+20)
    plt.grid(axis='y')
    sns.boxplot(data=dftotal, x='Time',y='Value', palette=("Set2"),flierprops = dict(markerfacecolor = '0.50', markersize = 2)).set(title= Topography[chan])
    
    
    p_fifteen = stats.ttest_ind(dfbase['Value'],dffifteen['Value'])[1]
    p_fortyfive = stats.ttest_ind(dfbase['Value'],dffortyfive['Value'])[1]
    
    p_values = [p_fortyfive, p_fifteen]
    decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=p_values, alpha=0.05, method='bonferroni')
    p_values = adj_pvals
    [p_fortyfive, p_fifteen] = adj_pvals
    print('T-test Base vs Fifteen: ' + str(p_fifteen))
    print('T-test Base vs Fortyfive: ' + str(p_fortyfive))
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

Total = pd.DataFrame (columns = ['Value', 'Time', 'Subject', 'Stimulation', 'Electrode','Normalized'])
Total = pd.concat([Total_20, Total_70, Total_Sham], ignore_index = True)
ymin=-30
ymax=20

plt.figure(fig, figsize= (15,15))
fig = fig+1
plt.ylim(ymin,ymax)
First = Total[np.isin(Total, ['FC3','FC1']).any(axis=1)]

sns.set_theme(style="whitegrid")
ax = sns.catplot(
    data = First, x = 'Time', y='Value',
    hue='Stimulation', capsize=.05, palette='magma', col = 'Electrode',
    kind = 'point', errorbar = 'sd', aspect=0.75
    )
ax.set(xlabel='Time', ylabel='MRBD(%)')



plt.figure(fig, figsize= (15,15))
fig = fig+1
plt.ylim(ymin,ymax)
Second = Total[np.isin(Total, ['C5','C3', 'C1']).any(axis=1)]

sns.set_theme(style="whitegrid")
ax = sns.catplot(
    data = Second, x = 'Time', y='Value',
    hue='Stimulation', capsize=.05, palette='magma', col = 'Electrode',
    kind = 'point', errorbar = 'sd',aspect=0.75
    )
ax.set(xlabel='Time', ylabel='MRBD(%)')


plt.figure(fig, figsize= (15,15))
fig = fig+1
plt.ylim(ymin,ymax)
Third = Total[np.isin(Total, ['CP5','CP3', 'CP1']).any(axis=1)]

sns.set_theme(style="whitegrid")
ax = sns.catplot(
    data = Third, x = 'Time', y='Value',
    hue='Stimulation', capsize=.05, palette='magma', col = 'Electrode',
    kind = 'point', errorbar = 'sd',aspect=0.75
    )
ax.set(xlabel='Time', ylabel='MRBD(%)')
plt.ylim(ymin,ymax)

#Test if baselines are different



Baseline_20 = Baseline_20.dropna()
Baseline_70 = Baseline_70.dropna()
Baseline_Sham = Baseline_Sham.dropna()
for chan in range(8):
    print (Topography[chan])
    p_70vs20 = stats.ttest_ind(Baseline_70[Topography[chan]],Baseline_20[Topography[chan]])[1]
    print('70 vs 20 ' + str(p_70vs20))
    p_70vsSham = stats.ttest_ind(Baseline_70[Topography[chan]],Baseline_Sham[Topography[chan]])[1]
    print('70 vs SHam ' + str(p_70vsSham))
    p_20vsSham = stats.ttest_ind(Baseline_20[Topography[chan]],Baseline_Sham[Topography[chan]])[1]
    print('20 vs Sham ' + str(p_20vsSham))


#Normalize data


Stim = ['20 Hz', '70 Hz', 'Sham']

grouped_electrodes = Total.groupby('Electrode')
Normalized = pd.DataFrame (columns = ['Value', 'Time', 'Stimulation', 'Electrode','Normalized'])
for chan in range(8):
    Electrode = grouped_electrodes.get_group(Topography[chan])
    grouped_session = Electrode.groupby('Stimulation')
    for ses in range(3):
        Session = grouped_session.get_group(Stim[ses])
        mean = Session.loc[Session['Time']== 'Baseline', 'Value'].mean()
        Session['Normalized'] = Session['Value']-(mean)
    
        Normalized = pd.concat([Normalized, Session], ignore_index = True)
    
ymin=-15
ymax=20
    
plt.figure(fig, figsize= (15,15))
fig = fig+1
plt.ylim(ymin,ymax)
First = Normalized[np.isin(Normalized, ['FC3','FC1']).any(axis=1)]

sns.set_theme(style="whitegrid")
ax = sns.catplot(
    data = First, x = 'Time', y='Normalized',
    hue='Stimulation', capsize=.05, palette='magma', col = 'Electrode',
    kind = 'point', errorbar = 'sd', aspect=0.75
    )
ax.set(xlabel='Time', ylabel='MRBD(%)\n Normalized to baseline')

plt.figure(fig, figsize= (15,15))
fig = fig+1
plt.ylim(ymin,ymax)
Second = Normalized[np.isin(Normalized, ['C5','C3', 'C1']).any(axis=1)]

sns.set_theme(style="whitegrid")
ax = sns.catplot(
    data = Second, x = 'Time', y='Normalized',
    hue='Stimulation', capsize=.05, palette='magma', col = 'Electrode',
    kind = 'point', errorbar = 'sd',aspect=0.75
    )
ax.set(xlabel='Time', ylabel='MRBD(%)\n Normalized to baseline')


plt.figure(fig, figsize= (15,15))
fig = fig+1
plt.ylim(ymin,ymax)
Third = Normalized[np.isin(Normalized, ['CP5','CP3', 'CP1']).any(axis=1)]

sns.set_theme(style="whitegrid")
ax = sns.catplot(
    data = Third, x = 'Time', y='Normalized',
    hue='Stimulation', capsize=.05, palette='magma', col = 'Electrode',
    kind = 'point', errorbar = 'sd',aspect=0.75
    )
ax.set(xlabel='Time', ylabel='MRBD(%)\n Normalized to baseline')
plt.ylim(ymin,ymax)













