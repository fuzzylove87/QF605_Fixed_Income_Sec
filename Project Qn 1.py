# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 22:45:56 2019

@author: Johnny Quek / Woon Tian Yong
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

data=pd.read_excel('IR Data.xlsx',sheet_name='OIS',header=0)
data_IRS=pd.read_excel('IR Data.xlsx',sheet_name='IRS',header=0)

df=pd.DataFrame(np.zeros((60,7)),columns=['Tenor','OIS','ON','DF','L', 'Fwd_L' ,'L_DF'])

data['Tenor']=data['Tenor'].apply(lambda x: (float(x[0:-1])/12 if x[-1] == 'm' else float((x[0:-1]))))
data_IRS['Tenor']=data_IRS['Tenor'].apply(lambda x: (float(x[0:-1])/12 if x[-1] == 'm' else float((x[0:-1]))))
df['Tenor'] = np.linspace(data['Tenor'].iloc[0], data['Tenor'].iloc[-1], num=(data['Tenor'].iloc[-1]*2))

df = df.set_index('Tenor')
data = data.set_index('Tenor')
data_IRS  = data_IRS.set_index('Tenor')

# Skipping 0.5y for OIS
df.merge(data, left_on='Tenor', right_on='Tenor')
df.merge(data_IRS, left_on='Tenor', right_on='Tenor')

df['OIS']=data['Rate']
df['L'] = data_IRS['Rate']

df = df.reset_index()

def OIS_Solver(df,ON,T,Gap=1):
    
    # df: DataFrame with relevant data
    # ON: The overnight rate, the variable we want to solve for
    # T: The tenor which you want to solve UP TO, assuming the Gap setting is constant
    # Gap: The gap between T and the starting point without given OIS rates (interpolation required)
    
    # Computing Fixed Leg PV
    
    DF_New=(1/(1+(ON/360)))**(T*360)
    
    #Get last known row #
    
    T_row_num = df[(df['Tenor']==T) & (df['OIS'].notnull())].index[0]
    
    if Gap>0:
        last_known_OIS_T = df[(df['Tenor']<T) & (df['OIS'].notnull())].index[-1]
        for i in range(Gap-1):
            df.iloc[last_known_OIS_T+i+1,3]=DF_New + (Gap-(i+1))*(1/Gap)*(df['DF'][last_known_OIS_T]-DF_New)
            df.iloc[last_known_OIS_T+i+1,2]=((df.iloc[last_known_OIS_T+i+1,3]**(-1/(df['Tenor'][last_known_OIS_T+i+1]*360)))-1)*360
    
    # sum only yearly DF since this a fixed annual OIS
    DF_Sum=df[df['Tenor'] % 1 == 0]['DF'].sum()
    
    Fix = (DF_Sum + DF_New)*df[df['Tenor'] == T]['OIS']
        
    # Computing Floating Leg PV
    
    Float_Sum=0
    Float_New=DF_New*(((1+(ON/360))**360)-1)
    
    for i in range(T_row_num):
        if df['Tenor'][i] % 1 ==0:
            F=df['DF'][i]*(((1+(df['ON'][i]/360))**360)-1)
            Float_Sum+=F
    
    Float = Float_Sum + Float_New
    
    return Fix - Float


def IRS_Solver(df,disc_f,T,coupon_period,Gap=1):
    
    # df: DataFrame with relevant data
        ## 0 - Tenor
        ## 1 - OIS
        ## 2 - ON
        ## 3 - DF
        ## 4 - L
        ## 5 - Fwd_L
        ## 6 - L_DF
        
    # L: The forward libor rate, the variable we want to solve for (i.e. if sent in T=2, L is L(2-coupon period,2))
    # T: The tenor which you want to solve UP TO, assuming the Gap setting is constant
    # Gap: The gap between T and the starting point without given OIS rates (interpolation required)
    
    # Computing Fixed Leg PV

    #Get last known row #
    T_row_num = df[(df['Tenor']==T) & (df['L'].notnull())].index[0]    
    
            
    DF_Sum=df[df['Tenor'] <= T ]['DF'].sum()
    
    Fix = coupon_period*(DF_Sum)*df[df['Tenor'] == T]['L']
        
    # Computing Floating Leg PV
    
    Float_Sum=0
    
    if Gap>0:
        last_known_IRS_T = df[(df['Tenor']<T) & (df['L'].notnull())].index[-1]
        # To interpolate for the gap
        for i in range(Gap-1):
            df.iloc[last_known_IRS_T+i+1,6]=disc_f + (Gap-(i+1))*(1/Gap)*(df['L_DF'][last_known_IRS_T]-disc_f)
            df.iloc[last_known_IRS_T+i+1,5]= ((df.iloc[last_known_IRS_T+i,6] - df.iloc[last_known_IRS_T+i+1,6])/
                                                df.iloc[last_known_IRS_T+i+1,6])/coupon_period
            
        FL_New = (((df[df['Tenor'] < T ]['L_DF'].values[-1] - disc_f)/(disc_f))/coupon_period)
    else:
        FL_New = (((1 - disc_f)/(disc_f))/coupon_period)
    

    Float_New= coupon_period * df[df['Tenor'] == T]['DF']* FL_New

    for i in range(T_row_num):
        F=coupon_period * df['DF'][i]*df['Fwd_L'][i]
        Float_Sum+=F
    
    Float = Float_Sum + Float_New
    
    return Fix - Float


def Par_Swap_Solver(fwd,swap,coupon_period=0.5):
    
    #Get swap FV
    S_maturity = fwd+swap
    float_leg = coupon_period * sum(df[(df['Tenor'] > fwd) & (df['Tenor'] <= S_maturity)]['Fwd_L'] * 
                                       df[(df['Tenor'] > fwd) & (df['Tenor'] <= S_maturity)]['DF'])
    disc_f = coupon_period * sum(df[(df['Tenor'] > fwd) & (df['Tenor'] <= S_maturity)]['DF'])
    
    return float_leg/disc_f


# Now solving for all Disc Factors (DF) and Overnight Rates (ON)
last_row = 0
for i in (df[(df['OIS'].notnull())]['Tenor']):
    row_num = df[(df['Tenor']==i)].index[0]
    Gap = row_num - last_row
    
    # TO GET OIS discount factors and ON rates
    df.iloc[row_num,2]=fsolve(lambda x: OIS_Solver(df,x,i,int(Gap)),0.0001)
    df.iloc[row_num,3]=(1/(1+(df.iloc[row_num,2]/360)))**(i*360)
    
    # TO GET LIBOR discount factors and fwd libor rates
    df.iloc[row_num,6]=fsolve(lambda x: IRS_Solver(df,x,i,0.5,int(Gap)),0.0001)
    
    if (row_num == 0):
        df.iloc[row_num,5]=((1 - df.iloc[row_num,6])/df.iloc[row_num,6])/0.5
    else:
        df.iloc[row_num,5]=((df.iloc[row_num-1,6] - df.iloc[row_num,6])/df.iloc[row_num,6])/0.5
        
    last_row = row_num
    

ps_df = pd.DataFrame([[1,1,Par_Swap_Solver(1,1)], [1,2,Par_Swap_Solver(1,2)], [1,3,Par_Swap_Solver(1,3)], [1,5,Par_Swap_Solver(1,5)], [1,10,Par_Swap_Solver(1,10)],
                      [5,1,Par_Swap_Solver(5,1)], [5,2,Par_Swap_Solver(5,2)], [5,3,Par_Swap_Solver(5,3)], [5,5,Par_Swap_Solver(5,5)], [5,10,Par_Swap_Solver(5,10)],
                      [10,1,Par_Swap_Solver(10,1)], [10,2,Par_Swap_Solver(10,2)], [10,3,Par_Swap_Solver(10,3)], [10,5,Par_Swap_Solver(10,5)], [10,10,Par_Swap_Solver(10,10)]
                      ],columns=['fwd','swap','par_swap_rate'])