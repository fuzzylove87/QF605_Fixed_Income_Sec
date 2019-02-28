# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 20:51:47 2019

@author: Wang Boyu
"""

import pandas as pd
import numpy as np
import scipy.stats as ss
from scipy.optimize import brentq
import matplotlib.pylab as plt
from scipy.optimize import least_squares
from Part_1 import df


#import data
data=pd.read_csv('Swaption.csv',usecols=range(13),index_col=['Expiry','Tenor'],skiprows=[16,17,18])

Swap=pd.read_excel('ATM.xlsx')

#df=pd.read_csv('E:/Github/QF605_Fixed_Income_Sec/df.csv')

Expiry=np.array(['1Y','5Y','10Y'])
Tenor=np.array(['1Y','2Y','3Y','5Y','10Y'])
s=np.array([-200,-150,-100,-50,-25,0,25,50,100,150,200])
Notion=10_000

#Present value of basic point(PVBP)
def PVBP(expiry,tenor,coupon_period=0.5):
    value=0.0
    stop=expiry+tenor
    time=np.arange(expiry+0.5,stop+0.1,0.5)
    for i in time:
        value=value+df[df['Tenor']==i].DF.values
    return value[0]*coupon_period

#Q1:DD Calibration
def DD_Model (F, K, sigma, T, Beta, TYPE):
    r=1
    if Beta==0:
        x_Star = (K-F)/(F*sigma*np.sqrt(T))
        Call=((F-K)*ss.norm.cdf(-x_Star,0,1)+
                             F*sigma*np.sqrt(T)*ss.norm.pdf(-x_Star,0,1))
        Put=((K-F)*ss.norm.cdf(x_Star,0,1)+
                             F*sigma*np.sqrt(T)*ss.norm.pdf(-x_Star,0,1))
    else:
        F=F/Beta
        K=K+((1-Beta))*F
        sigma=Beta*sigma
        d1 = (np.log(F/K)+(sigma**2)/2*T)/(sigma*np.sqrt(T))
        d2 = (np.log(F/K)-(sigma**2)/2*T)/(sigma*np.sqrt(T))
        Call = (F*ss.norm.cdf(d1,0,1)-K*ss.norm.cdf(d2,0,1))
        Put =  (K*ss.norm.cdf(-d2,0,1)-F*ss.norm.cdf(-d1,0,1))
    if TYPE == 'Put':
        return Put
    else:
        return Call
    
def DD_Calibrate_Price(x,ATM,strikes,T,TN,vols):
    err=0.0
    for i,vol in zip(range(len(strikes)), vols):
        price1 = Notion*DD_Model(ATM, strikes[i], vol, T, 1,'Call') #Beta=1 using Black76
        price2 = Notion*DD_Model(ATM, strikes[i], x[0], T, x[1], 'Call')
        err += (price1 - price2)**2
        #plt.scatter(strikes[i],price1,color='green')
    return (PVBP(T,TN,0.5)**2)*err

initialGuess_DD=[0.5,0.2]
Cal_Beta=pd.DataFrame(np.zeros((3,5)),index=Expiry,columns=Tenor)
Cal_Sigma=pd.DataFrame(np.zeros((3,5)),index=Expiry,columns=Tenor)

for ex in Expiry:
    for te in Tenor:             #Set sigma,T,ATM,strikes
        sigma=data.loc[(ex,te), 'ATM']/100
        T=int(ex[:-1])
        TN=int(te[:-1])
        ATM=Swap[te][ex] #ATM and K changes
        strikes=s*0.0001+ATM
        res = least_squares(lambda x: DD_Calibrate_Price(x,
                                          ATM,
                                          strikes,
                                          T,   
                                          TN,
                                          data.loc[ex,te].values/100),
                                          initialGuess_DD,
                                          bounds=([0,0],[np.inf,1]))
        Cal_Beta[te][ex]= res.x[1]
        Cal_Sigma[te][ex]=res.x[0]
        
print('=========Calibrated Displaced-Difusion Model Parameters==========')
print('Sigma')
print(Cal_Sigma)

print('Beta')
print(Cal_Beta)


#Q2 SABR Calibration
def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    if F == K:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
        zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom
    return sabrsigma

def sabrcalibration(x, strikes, vols, F, T):
    err = 0.0
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T,
                           x[0], 0.9, x[1], x[2]))**2
    return err

# Calibration
SABR_Alpha=pd.DataFrame(np.zeros((3,5)),index=Expiry,columns=Tenor)
SABR_Rho=pd.DataFrame(np.zeros((3,5)),index=Expiry,columns=Tenor)
SABR_Nu=pd.DataFrame(np.zeros((3,5)),index=Expiry,columns=Tenor)
initialGuess_sabr=[0.1,-0.5,0.5]

for ex in Expiry:
    for te in Tenor:             #Set sigma,T,ATM,strikes
        sigma=data.loc[(ex,te), 'ATM']/100
        T=int(ex[:-1])
        ATM=Swap[te][ex] #ATM and K changes
        strikes=s*0.0001+ATM
        res=least_squares(lambda x: sabrcalibration(x,
                                           strikes,
                                           data.loc[ex,te].values/100,# [1,1] change to [ex,te]
                                           ATM,
                                           T),
                          initialGuess_sabr,
                          bounds=([0,-1,0],[np.inf,1,np.inf]))
        SABR_Alpha[te][ex]=res.x[0]
        SABR_Rho[te][ex]=res.x[1]
        SABR_Nu[te][ex]=res.x[2]

print('=========Calibrated SABR Model Parameters=========')
print('===Alpha===')
print(SABR_Alpha)
print('===Rho===')
print(SABR_Rho)
print('===Nu===')
print(SABR_Nu)

#Q3 Price the following swaptions:
#=========payer 2y x 10y K = 1%; 2%; 3%; 4%; 5%; 6%; 7%; 8%=============

def Black76Pay(F, K, T, sigma):
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return (F*ss.norm.cdf(d1) - K*ss.norm.cdf(d2))

def Black76Rec(F, K, T, sigma):
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return (K*ss.norm.cdf(-d2) - F*ss.norm.cdf(-d1))

def interpol(x1,y1,x2,y2,x):
    return y1+(x-x1)*((y2-y1)/(x2-x1))

#Parameters
strike=np.arange(0.01,0.081,0.01)
T,TN=2,10
ATM=interpol(1,0.038436,5,0.043676,2)

#Interpolate SABR
alpha=interpol(1,SABR_Alpha['10Y']['1Y'],5,SABR_Alpha['10Y']['5Y'],2)
rho=interpol(1,SABR_Rho['10Y']['1Y'],5,SABR_Rho['10Y']['5Y'],2)
nu=interpol(1,SABR_Nu['10Y']['1Y'],5,SABR_Nu['10Y']['5Y'],2)
value1_SABR=[PVBP(T,TN,0.5)*Notion*Black76Pay(ATM,x,T,SABR(ATM,x,T,alpha,0.9,rho,nu)) for x in strike]

#Interpolate DD 
sigma=interpol(1,Cal_Sigma['10Y']['1Y'],5,Cal_Sigma['10Y']['5Y'],2)
Beta=interpol(1,Cal_Beta['10Y']['1Y'],5,Cal_Beta['10Y']['5Y'],2)
value1_DD=[PVBP(T,TN,0.5)*Notion*DD_Model (ATM, x, sigma, T, Beta, 'Call') for x in strike ]

value1=pd.DataFrame(value1_SABR,index=strike,columns=['SABR'])
value1.index.name='Strikes'
value1['Displaced-Diffusion']=value1_DD

#=========receiver 8y x 10y K = 1%; 2%; 3%; 4%; 5%; 6%;7%;8%=============
strike=np.arange(0.01,0.081,0.01)
T,TN=8,10
ATM=interpol(5,0.043676,10,0.053545,8)

#SABR Pricing
alpha=interpol(5,SABR_Alpha['10Y']['5Y'],10,SABR_Alpha['10Y']['10Y'],8)
rho=interpol(5,SABR_Rho['10Y']['5Y'],10,SABR_Rho['10Y']['10Y'],8)
nu=interpol(5,SABR_Nu['10Y']['5Y'],10,SABR_Nu['10Y']['10Y'],8)

value2_SABR=[PVBP(T,TN,0.5)*Notion*Black76Rec(ATM,x,T,SABR(ATM,x,T,alpha,0.9,rho,nu)) for x in strike]

#DD Pricing
sigma=interpol(5,Cal_Sigma['10Y']['5Y'],10,Cal_Sigma['10Y']['10Y'],8)
Beta=interpol(5,Cal_Beta['10Y']['5Y'],10,Cal_Beta['10Y']['10Y'],8)

value2_DD=[PVBP(T,TN,0.5)*Notion*DD_Model (ATM, x, sigma, T, Beta, 'Put') for x in strike ]

value2=pd.DataFrame(value2_SABR,index=strike,columns=['SABR'])
value2.index.name='Strikes'
value2['Displaced-Diffusion']=value2_DD

print('=========Price of 2y x 10y payer swaption=========')
print(value1)

print('=========Price of 8y x 10y receiver swaption=========')
print(value2)

#====================plot and output================
def impliedVolatility(S, K, price, T, Beta):
    impliedVol = brentq(lambda x: price - 
                        DD_Model(S, K,  x, T, 1, 'Call'),
                        1e-6, 1)
    return impliedVol

for ex in Expiry:
    fig=plt.figure(figsize=(5,5))
    ax=plt.axes()
    ax=plt.title('Tenor=%s' %te)
    for te,i in zip(Tenor,range(5)):
        ax=plt.subplot(551+i)
        ATM=Swap[te][ex] #ATM and K changes
        strikes=s*0.0001+ATM
        T=int(ex[:-1])
        ax=plt.plot(strikes,data.loc[ex,te].values/100,'go',label='Market Vol')
        ax=plt.plot(strikes,
                    [SABR(ATM,x,T,SABR_Alpha.loc[ex,te],0.9,SABR_Rho.loc[ex,te],SABR_Nu.loc[ex,te]) for x in strikes],
                    'k',label='SABR Vol')
        
        price_DD=[DD_Model(ATM,x, Cal_Sigma[te][ex], T,Cal_Beta[te][ex] ,'Call') for i,x in zip(range(len(strikes)),strikes)]
        imp=[impliedVolatility(ATM, x, p, T, Cal_Beta[te][ex]) for x,p in zip(strikes,price_DD)]
        ax=plt.plot(strikes,imp,color='blue',label='DD Model Implied Vol')
        #Graph modify
        ax=plt.title((ex,te))
        ax=plt.legend()
    plt.savefig('%s'%ex,dpi=100)
    plt.show()

SABR_Alpha.to_csv('Alpha.csv')
SABR_Rho.to_csv('Rho.csv')
SABR_Nu.to_csv('Nu.csv')


value1.to_csv('Value1.csv')
value2.to_csv('Value2.csv')






















