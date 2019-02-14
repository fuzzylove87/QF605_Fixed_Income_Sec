# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 01:24:50 2019
Project Qn.1
@author: Johnny
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.interpolate as interpolate

#Get OIS rates from files
OIS_rates = pd.read_excel('IR Data.xlsx',sheet_name='OIS',usecols = "A:C",header=0)

#Get LIBOR rates from files
IRS_rates = pd.read_excel('IR Data.xlsx',sheet_name='IRS',usecols = "A:C",header=0)

#set tenor to year format
OIS_rates['Tenor_Y'] = OIS_rates['Tenor'].apply(lambda x: (float(x[0:-1])/12 if x[-1] == 'm' else float((x[0:-1]))))
IRS_rates['Tenor_Y'] = IRS_rates['Tenor'].apply(lambda x: (float(x[0:-1])/12 if x[-1] == 'm' else float((x[0:-1]))))
OIS_rates['discount_factors'] = np.nan
IRS_rates['discount_factors'] = np.nan      
#######################################
############# PART 1 ##################
#######################################
OIS_discount_factor = np.zeros((1, 2))
for i in range(len(OIS_rates['Tenor_Y'])):
    if OIS_rates['Tenor_Y'].iloc[i] < 1 :
        #TODO: refactor the discount factors to include daycount convention
        OIS_rates['discount_factors'].iloc[i] = 1/(1+ OIS_rates['Rate'].iloc[i]*OIS_rates['Tenor_Y'].iloc[i])
    else:
        gap = int(OIS_rates['Tenor_Y'].iloc[i] - OIS_rates['Tenor_Y'].iloc[i-1])
        if (gap >1):
            #TODO: to calculate for the next tenor first
            coeff = (sum(range(gap))/gap) 
            preceding_df = OIS_discount_factor[-1][1]
            next_df = (1 - OIS_rates['Rate'].iloc[i]*((OIS_discount_factor.sum(axis=0)[1]) + coeff * preceding_df))/( 1 + (1+coeff)*OIS_rates['Rate'].iloc[i])
            df = np.array([(OIS_rates['Tenor_Y'].iloc[i],next_df)])
            OIS_discount_factor = np.concatenate((OIS_discount_factor,df))

            #TODO: to add in the gap years with interpolation
            for g in range(gap-1):
                df = np.array([(OIS_rates['Tenor_Y'].iloc[i-1] + g + 1,preceding_df - ((preceding_df-next_df)/gap*(g+1)))])
                OIS_discount_factor = np.concatenate((OIS_discount_factor,df))
                #print(preceding_df,next_df ,preceding_df - ((preceding_df-next_df)/gap*(g+1)))
            
            # Sort it by ascending order for later use
            OIS_discount_factor = OIS_discount_factor[np.argsort(OIS_discount_factor[:, 0])]
            
        else:
            df = np.array([(OIS_rates['Tenor_Y'].iloc[i],(1 - OIS_rates['Rate'].iloc[i]*((OIS_discount_factor.sum(axis=0)[1])))/(1 + OIS_rates['Rate'].iloc[i]))])
            OIS_discount_factor = np.concatenate((OIS_discount_factor,df))
 
#######################################
############# PART 2 ##################
#######################################
 
LIBOR_discount_factor = np.zeros((1, 2))
for i in range(len(IRS_rates['Tenor_Y'])):

        gap = int(IRS_rates['Tenor_Y'].iloc[i] - IRS_rates['Tenor_Y'].iloc[i-1])
        coupon_period = 0.5
        #TODO: to set gap to be based on the coupon period
        if (gap > coupon_period):
            #TODO: to calculate for the next tenor first
            coeff = (sum(range(int(gap/coupon_period)))/int(gap/coupon_period)) 
            preceding_df = LIBOR_discount_factor[-1][1]
            next_df = (1 - coupon_period*IRS_rates['Rate'].iloc[i]*((LIBOR_discount_factor.sum(axis=0)[1]) + coeff * preceding_df))/( 1 + (1+coeff)*coupon_period*IRS_rates['Rate'].iloc[i])
            df = np.array([(IRS_rates['Tenor_Y'].iloc[i],next_df)])
            LIBOR_discount_factor = np.concatenate((LIBOR_discount_factor,df))

            #TODO: to add in the gap years with interpolation
            for g in range(int(gap/coupon_period-1)):
                df = np.array([(IRS_rates['Tenor_Y'].iloc[i-1] + (g+1)*coupon_period ,preceding_df - ((preceding_df-next_df)/(gap/coupon_period*(g+1))))])
                LIBOR_discount_factor = np.concatenate((LIBOR_discount_factor,df))
                print(preceding_df,next_df ,preceding_df - ((preceding_df-next_df)/(gap/coupon_period*(g+1))))
            
            # Sort it by ascending order for later use
            LIBOR_discount_factor = LIBOR_discount_factor[np.argsort(LIBOR_discount_factor[:, 0])]
            
        else:
            df = np.array([(IRS_rates['Tenor_Y'].iloc[i],(1 - coupon_period* IRS_rates['Rate'].iloc[i]*((LIBOR_discount_factor.sum(axis=0)[1])))/(1 + coupon_period*IRS_rates['Rate'].iloc[i]))])
            LIBOR_discount_factor = np.concatenate((LIBOR_discount_factor,df))

      
#interpolating linearly for OIS rates
#TODO: to pull in automatically depending on the floating leg convention
OIS_interplt_rates = pd.DataFrame( columns=['tenor','discount_factors'])
OIS_interplt_rates['tenor'] = np.linspace(OIS_rates['Tenor_Y'].iloc[0], OIS_rates['Tenor_Y'].iloc[-1], num=(OIS_rates['Tenor_Y'].iloc[-1]*2))
interp = interpolate.interp1d(OIS_rates['Tenor_Y'], OIS_rates['discount_factors'])
OIS_interplt_rates['discount_factors'] = OIS_interplt_rates['tenor'].apply(lambda x: float(interp(x)))


IRS_rates['Tenor_Y'] = IRS_rates['Tenor'].apply(lambda x: (float(x[0:-1])/12 if x[-1] == 'm' else float((x[0:-1]))))
IRS_rates['discount_factors'] = OIS_rates['discount_factors']
