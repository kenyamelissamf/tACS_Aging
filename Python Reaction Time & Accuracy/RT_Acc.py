# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:10:29 2023

@author: Kenya
"""

import numpy as np
import pandas as pd

# import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import scipy.io
from scipy import stats
import statsmodels.stats.multitest

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

Error_20= pd.DataFrame(columns = [ 'Error', 'Time', 'Stimulation'])
Error_70= pd.DataFrame(columns = [ 'Error', 'Time', 'Stimulation'])
Error_Sham= pd.DataFrame(columns = [ 'Error', 'Time', 'Stimulation'])


## LOOP FOR ERROR

#Loop for subjects
for u in range(14):
    sub = SubjectID[u+1] 
    dfall2 = pd.DataFrame (columns = ['Values'])
    #Loop for sessions (1-3)
    
    for y in range(3):
        dfall = pd.DataFrame (columns = ['Values'])
        
        #Loop for recordings (1-4)
        for i in range(4):
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
    
    
    plt.figure(fig)
    subp = 130
    plt.figure(fig).set_figwidth(15)
    
    for y in range(3):
        dfnew = pd.DataFrame(columns = ['Error', 'Time'])
            
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
        
        if Dict[sub][y] == '20 Hz':
            Error_20 = pd.concat([dfnew,Error_20])
            Error_20['Stimulation'] = '20 Hz'
        elif Dict[sub][y] == '70 Hz':
            Error_70 = pd.concat([dfnew,Error_70])
            Error_70['Stimulation'] = '70 Hz'
        elif Dict[sub][y] == 'Sham':
            Error_Sham = pd.concat([dfnew,Error_Sham])
            Error_Sham['Stimulation'] = 'Sham'

        
        #Statistics
        Base_Stats = dfnew.loc[dfnew['Time'] == 'Baseline']
        tACS_Stats = dfnew.loc[dfnew['Time'] == 'tACS']
        Fifteen_Stats = dfnew.loc[dfnew['Time'] == '15min']
        Fortyfive_Stats = dfnew.loc[dfnew['Time'] == '45min']
        
        p_tacs = stats.ttest_ind(Base_Stats['Error'],tACS_Stats['Error'])[1]
        p_fifteen = stats.ttest_ind(Base_Stats['Error'],Fifteen_Stats['Error'])[1]
        p_fortyfive = stats.ttest_ind(Base_Stats['Error'],Fortyfive_Stats['Error'])[1]
        
        p_values = [p_fortyfive, p_fifteen, p_tacs]
        decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=p_values, alpha=0.05, method='bonferroni')
        p_values = adj_pvals
        [p_fortyfive, p_fifteen, p_tacs] = adj_pvals
        print('T-test Error Base vs tACS: ' + str(p_tacs))
        print('T-test Error Base vs Fifteen: ' + str(p_fifteen))
        print('T-test Error Base vs Fortyfive: ' + str(p_fortyfive))

        #Plotting in seaborn boxplot
        plt.subplot(subp+y+1)
        plt.ylim(ymin-0.01,ymax+0.07)
        sns.boxplot(data=dfnew, x='Time',y='Error',flierprops = dict(markerfacecolor = '0.50', markersize = 2)).set(title= 'Accuracy ' + sub + ' ' + Dict[sub][y])
        
        # Get the y-axis limits
        y_range = ymax - ymin
        
        levels = [3,2,1]
        #Plotting significance bars
        for n in range(3):
            if p_values[n] < 0.05:
                level = levels[n]
                x1 = 0
                x2 = 3-n
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
            

        #Printing when a subject for each session is done plotting
        print(sub + ' Error ' + Dict[sub][y] + ' Done')
    fig += 1
    
del dfnew
del dfall2

Error_total = pd.DataFrame (columns = ['Error', 'Time', 'Stimulation'])
Error_total = pd.concat([Error_20, Error_70, Error_Sham], ignore_index = True)

#20 Hz Group-level

print('20 Hz Group-level')

plt.figure(fig).set_figwidth(15)
fig = fig+1
subp = 130

ymax = Error_total['Error'].max()
ymin = Error_total['Error'].min()

plt.subplot(subp+1)
plt.ylim(ymin-0.05,ymax+0.16)
sns.boxplot(data=Error_20, x='Time',y='Error', palette=("Set2"), flierprops = dict(markerfacecolor = '0.50', markersize = 1)).set(title= '20 Hz')

p_tACS = stats.ttest_ind(Error_20.loc[Error_20['Time'] == 'Baseline']['Error'],Error_20.loc[Error_20['Time'] == 'tACS']['Error'])[1]
p_fifteen = stats.ttest_ind(Error_20.loc[Error_20['Time'] == 'Baseline']['Error'],Error_20.loc[Error_20['Time'] == '15min']['Error'])[1]
p_fortyfive = stats.ttest_ind(Error_20.loc[Error_20['Time'] == 'Baseline']['Error'],Error_20.loc[Error_20['Time'] == '45min']['Error'])[1]


p_values = [p_fortyfive, p_fifteen, p_tACS]
decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=p_values, alpha=0.05, method='bonferroni')
p_values = adj_pvals
[p_fortyfive, p_fifteen, p_tacs] = adj_pvals
print('T-test Base vs tACS: ' + str(p_tACS))
print('T-test Base vs Fifteen: ' + str(p_fifteen))
print('T-test Base vs Fortyfive: ' + str(p_fortyfive))

# Get the y-axis limits
y_range = ymax - ymin

levels = [3,2,1]
#Plotting significance bars
for n in range(3):
    if p_values[n] < 0.05:
        level = levels[n]
        x1 = 0
        x2 = 3-n
        # Plot the bar
        bar_height = (y_range * 0.1 * level) + ymax
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
    
    
#70 Hz Group-level

print('70 Hz Group-level')

plt.subplot(subp+2)
plt.ylim(ymin-0.05,ymax+0.16)
sns.boxplot(data=Error_70, x='Time',y='Error', palette=("Set2"),flierprops = dict(markerfacecolor = '0.50', markersize = 1)).set(title= '70 Hz',ylabel=None)

p_tACS = stats.ttest_ind(Error_70.loc[Error_70['Time'] == 'Baseline']['Error'],Error_70.loc[Error_70['Time'] == 'tACS']['Error'])[1]
p_fifteen = stats.ttest_ind(Error_70.loc[Error_70['Time'] == 'Baseline']['Error'],Error_70.loc[Error_70['Time'] == '15min']['Error'])[1]
p_fortyfive = stats.ttest_ind(Error_70.loc[Error_70['Time'] == 'Baseline']['Error'],Error_70.loc[Error_70['Time'] == '45min']['Error'])[1]


p_values = [p_fortyfive, p_fifteen, p_tACS]
decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=p_values, alpha=0.05, method='bonferroni')
p_values = adj_pvals
[p_fortyfive, p_fifteen, p_tacs] = adj_pvals
print('T-test Base vs tACS: ' + str(p_tACS))
print('T-test Base vs Fifteen: ' + str(p_fifteen))
print('T-test Base vs Fortyfive: ' + str(p_fortyfive))

# Get the y-axis limits
y_range = ymax - ymin

levels = [3,2,1]
#Plotting significance bars
for n in range(3):
    if p_values[n] < 0.05:
        level = levels[n]
        x1 = 0
        x2 = 3-n
        # Plot the bar
        bar_height = (y_range * 0.1 * level) + ymax
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


#Sham Group-level

print('Sham Hz Group-level')

plt.subplot(subp+3)
plt.ylim(ymin-0.05,ymax+0.16)
sns.boxplot(data=Error_Sham, x='Time',y='Error', palette=("Set2"),flierprops = dict(markerfacecolor = '0.50', markersize = 1)).set(title= 'Sham', ylabel=None)

p_tACS = stats.ttest_ind(Error_Sham.loc[Error_Sham['Time'] == 'Baseline']['Error'],Error_Sham.loc[Error_Sham['Time'] == 'tACS']['Error'])[1]
p_fifteen = stats.ttest_ind(Error_Sham.loc[Error_Sham['Time'] == 'Baseline']['Error'],Error_Sham.loc[Error_Sham['Time'] == '15min']['Error'])[1]
p_fortyfive = stats.ttest_ind(Error_Sham.loc[Error_Sham['Time'] == 'Baseline']['Error'],Error_Sham.loc[Error_Sham['Time'] == '45min']['Error'])[1]


p_values = [p_fortyfive, p_fifteen, p_tACS]
decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=p_values, alpha=0.05, method='bonferroni')
p_values = adj_pvals
[p_fortyfive, p_fifteen, p_tacs] = adj_pvals
print('T-test Base vs tACS: ' + str(p_tACS))
print('T-test Base vs Fifteen: ' + str(p_fifteen))
print('T-test Base vs Fortyfive: ' + str(p_fortyfive))

# Get the y-axis limits
y_range = ymax - ymin

levels = [3,2,1]
#Plotting significance bars
for n in range(3):
    if p_values[n] < 0.05:
        level = levels[n]
        x1 = 0
        x2 = 3-n
        # Plot the bar
        bar_height = (y_range * 0.1 * level) + ymax
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

######################################################################################################################
#LOOP FOR REACTION TIME

######################################################################################################################
#Loop for subjects   

RT_20= pd.DataFrame(columns = [ 'Reaction Time', 'Time', 'Stimulation'])
RT_70= pd.DataFrame(columns = [ 'Reaction Time', 'Time', 'Stimulation'])
RT_Sham= pd.DataFrame(columns = [ 'Reaction Time', 'Time', 'Stimulation']) 

for u in range(14):
    sub = SubjectID[u+1] 
    dfall2 = pd.DataFrame (columns = ['Values'])
   
    #Loop for sessions (1-3)
    
    for y in range(3):
        dfall = pd.DataFrame (columns = ['Values'])
        for i in range(4):
            #Extracting data to a new dataframe
            df1 = pd.DataFrame(columns = ['Values'])
            df1['Values'] = RTdf[sub][0][0,y][0,i].tolist()[4:-1]
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
    
    plt.figure(fig)
    subp = 130
    plt.figure(fig).set_figwidth(15)
    
    
    for y in range(3):
        dfnew = pd.DataFrame(columns = ['Reaction Time', 'Time'])
            
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
            #Prinitng which outliers were removed
            for k in range(indexes[-1]+1):
                if df['Reaction Time'][k] > condition:
                    print('Reaction Time Removed outlier (Positive): Trial #' + str(k+5) + ' Value: ' + str(df['Reaction Time'][k]) + ' ' + sub + ' ' + Dict[sub][y] + ' ' + Time[i] )
            for k in range(indexes[-1]+1):
                if df['Reaction Time'][k] < condition2:
                    print('Reaction Time Removed outlier (Negative): Trial #' + str(k+5) + ' Value: ' + str(df['Reaction Time'][k]) + ' ' + sub + ' ' + Dict[sub][y] + ' ' + Time[i] )
            
            #Removing outliers
            df22 = df.loc[df['Reaction Time'] < condition]
            df2 = df22.loc[df22['Reaction Time'] > condition2]
                
            #Concatenating all time dataframes into a single one for each session
            dfnew = pd.concat([dfnew,df2], ignore_index=True)
            del df
            
        
        #Converting to numpy array
        dfnew['Reaction Time'] = np.array(dfnew['Reaction Time'], dtype=np.float64)
        
        if Dict[sub][y] == '20 Hz':
            RT_20 = pd.concat([dfnew,RT_20])
            RT_20['Stimulation'] = '20 Hz'
        elif Dict[sub][y] == '70 Hz':
            RT_70 = pd.concat([dfnew,RT_70])
            RT_70['Stimulation'] = '70 Hz'
        elif Dict[sub][y] == 'Sham':
            RT_Sham = pd.concat([dfnew,RT_Sham])
            RT_Sham['Stimulation'] = 'Sham'
        
        #Statistics
        Base_Stats = dfnew.loc[dfnew['Time'] == 'Baseline']
        tACS_Stats = dfnew.loc[dfnew['Time'] == 'tACS']
        Fifteen_Stats = dfnew.loc[dfnew['Time'] == '15min']
        Fortyfive_Stats = dfnew.loc[dfnew['Time'] == '45min']
        
        p_tacs = stats.ttest_ind(Base_Stats['Reaction Time'],tACS_Stats['Reaction Time'])[1]
        p_fifteen = stats.ttest_ind(Base_Stats['Reaction Time'],Fifteen_Stats['Reaction Time'])[1]
        p_fortyfive = stats.ttest_ind(Base_Stats['Reaction Time'],Fortyfive_Stats['Reaction Time'])[1]
        
        p_values = [p_fortyfive, p_fifteen, p_tacs]
        decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=p_values, alpha=0.05, method='bonferroni')
        p_values = adj_pvals
        [p_fortyfive, p_fifteen, p_tacs] = adj_pvals
        print('T-test RT Base vs tACS: ' + str(p_tacs))
        print('T-test RT Base vs Fifteen: ' + str(p_fifteen))
        print('T-test RT Base vs Fortyfive: ' + str(p_fortyfive))

        #Plotting in seaborn boxplot
        plt.subplot(subp+y+1)
        plt.ylim(ymin-0.01,ymax+0.35)
        sns.boxplot(data=dfnew, x='Time',y='Reaction Time',flierprops = dict(markerfacecolor = '0.50', markersize = 2)).set(title= 'Reaction Time ' + sub + ' ' + Dict[sub][y])
        
        # Get the y-axis limits
        y_range = ymax - ymin
        
        levels = [3,2,1]
        #Plotting significance bars
        for n in range(3):
            if p_values[n] < 0.05:
                level = levels[n]
                x1 = 0
                x2 = 3-n
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

        
        #Printing when a subject for each session is done plotting
        print(sub + ' Reaction Time ' + Dict[sub][y] + ' Done')
    fig += 1

RT_total = pd.DataFrame (columns = ['Reaction Time', 'Time', 'Stimulation'])
RT_total = pd.concat([RT_20, RT_70, RT_Sham], ignore_index = True)

#20 Hz Group-level

print('20 Hz Group-level')

plt.figure(fig).set_figwidth(15)
fig = fig+1
subp = 130

ymax = RT_Sham['Reaction Time'].max()
ymin = RT_Sham['Reaction Time'].min()

plt.subplot(subp+1)
plt.ylim(ymin-0.1,ymax+0.7)
sns.boxplot(data=RT_20, x='Time',y='Reaction Time', palette=("Set2"),flierprops = dict(markerfacecolor = '0.50', markersize = 2)).set(ylabel='Reaction Time (sec)')

p_tACS = stats.ttest_ind(RT_20.loc[RT_20['Time'] == 'Baseline']['Reaction Time'],RT_20.loc[RT_20['Time'] == 'tACS']['Reaction Time'])[1]
p_fifteen = stats.ttest_ind(RT_20.loc[RT_20['Time'] == 'Baseline']['Reaction Time'],RT_20.loc[RT_20['Time'] == '15min']['Reaction Time'])[1]
p_fortyfive = stats.ttest_ind(RT_20.loc[RT_20['Time'] == 'Baseline']['Reaction Time'],RT_20.loc[RT_20['Time'] == '45min']['Reaction Time'])[1]


p_values = [p_fortyfive, p_fifteen, p_tACS]
decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=p_values, alpha=0.05, method='bonferroni')
p_values = adj_pvals
[p_fortyfive, p_fifteen, p_tacs] = adj_pvals
print('T-test Base vs tACS: ' + str(p_tACS))
print('T-test Base vs Fifteen: ' + str(p_fifteen))
print('T-test Base vs Fortyfive: ' + str(p_fortyfive))

# Get the y-axis limits
y_range = ymax - ymin

levels = [3,2,1]
#Plotting significance bars
for n in range(3):
    if p_values[n] < 0.05:
        level = levels[n]
        x1 = 0
        x2 = 3-n
        # Plot the bar
        bar_height = (y_range * 0.1 * level) + ymax
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
    
    
#70 Hz Group-level

print('70 Hz Group-level')

plt.subplot(subp+2)
plt.ylim(ymin-0.1,ymax+0.7)
sns.boxplot(data=RT_70, x='Time',y='Reaction Time', palette=("Set2"),flierprops = dict(markerfacecolor = '0.50', markersize = 2)).set(ylabel=None)

p_tACS = stats.ttest_ind(RT_70.loc[RT_70['Time'] == 'Baseline']['Reaction Time'],RT_70.loc[RT_70['Time'] == 'tACS']['Reaction Time'])[1]
p_fifteen = stats.ttest_ind(RT_70.loc[RT_70['Time'] == 'Baseline']['Reaction Time'],RT_70.loc[RT_70['Time'] == '15min']['Reaction Time'])[1]
p_fortyfive = stats.ttest_ind(RT_70.loc[RT_70['Time'] == 'Baseline']['Reaction Time'],RT_70.loc[RT_70['Time'] == '45min']['Reaction Time'])[1]


p_values = [p_fortyfive, p_fifteen, p_tACS]
decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=p_values, alpha=0.05, method='bonferroni')
p_values = adj_pvals
[p_fortyfive, p_fifteen, p_tacs] = adj_pvals
print('T-test Base vs tACS: ' + str(p_tACS))
print('T-test Base vs Fifteen: ' + str(p_fifteen))
print('T-test Base vs Fortyfive: ' + str(p_fortyfive))

# Get the y-axis limits
y_range = ymax - ymin

levels = [3,2,1]
#Plotting significance bars
for n in range(3):
    if p_values[n] < 0.05:
        level = levels[n]
        x1 = 0
        x2 = 3-n
        # Plot the bar
        bar_height = (y_range * 0.1 * level) + ymax
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


#Sham Group-level

print('Sham Hz Group-level')

plt.subplot(subp+3)
plt.ylim(ymin-0.1,ymax+0.7)
sns.boxplot(data=RT_Sham, x='Time',y='Reaction Time', palette=("Set2"),flierprops = dict(markerfacecolor = '0.50', markersize = 2)).set( ylabel=None)

p_tACS = stats.ttest_ind(RT_Sham.loc[RT_Sham['Time'] == 'Baseline']['Reaction Time'],RT_Sham.loc[RT_Sham['Time'] == 'tACS']['Reaction Time'])[1]
p_fifteen = stats.ttest_ind(RT_Sham.loc[RT_Sham['Time'] == 'Baseline']['Reaction Time'],RT_Sham.loc[RT_Sham['Time'] == '15min']['Reaction Time'])[1]
p_fortyfive = stats.ttest_ind(RT_Sham.loc[RT_Sham['Time'] == 'Baseline']['Reaction Time'],RT_Sham.loc[RT_Sham['Time'] == '45min']['Reaction Time'])[1]


p_values = [p_fortyfive, p_fifteen, p_tACS]
decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=p_values, alpha=0.05, method='bonferroni')
p_values = adj_pvals
[p_fortyfive, p_fifteen, p_tacs] = adj_pvals
print('T-test Base vs tACS: ' + str(p_tACS))
print('T-test Base vs Fifteen: ' + str(p_fifteen))
print('T-test Base vs Fortyfive: ' + str(p_fortyfive))

# Get the y-axis limits
y_range = ymax - ymin

levels = [3,2,1]
#Plotting significance bars
for n in range(3):
    if p_values[n] < 0.05:
        level = levels[n]
        x1 = 0
        x2 = 3-n
        # Plot the bar
        bar_height = (y_range * 0.1 * level) + ymax
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

ymin=0.14
ymax=0.19

sns.set_theme(style="whitegrid")
sns.catplot(
    data = Error_total, x = 'Time', y='Error',
    hue='Stimulation', capsize=.05, palette='magma',
    kind = 'point', errorbar = 'sd', aspect=1
    ).set(xlabel='Time', ylabel='Error', title ='Group level')
plt.ylim(ymin,ymax)

ymin= 0.5
ymax=0.65

sns.set_theme(style="whitegrid")
sns.catplot(
    data = RT_total, x = 'Time', y='Reaction Time',
    hue='Stimulation', capsize=.05, palette='magma',
    kind = 'point', errorbar = 'sd',aspect=0.9
    ).set(xlabel='Time', ylabel='Reaction Time', title = 'Group level')
plt.ylim(ymin,ymax)




