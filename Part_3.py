# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:35:11 2019

@author: Jin Weiguo, Kim ChanJung
"""

import numpy as np
import pandas as pd
import Part_1
from scipy.stats import norm
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

# Define necessary functions
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

def Black76Pay(F, K, T, sigma):
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return (F*norm.cdf(d1) - K*norm.cdf(d2))

def Black76Rec(F, K, T, sigma):
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return (K*norm.cdf(-d2) - F*norm.cdf(-d1))

def IRR(K,N,n,day):
    NOP = int((N-n)/day)
    list=[day/(1+day*K)**i for i in range(1,NOP+1)]
    return np.sum(list)

def IRR1prime(K,N,n,day):
    NOP = int((N-n)/day)
    list=[day**2*(-i)*(1+day*K)**(-i-1) for i in range(1,NOP+1)]
    return np.sum(list)

def IRR2prime(K,N,n,day):
    NOP = int((N-n)/day)
    list=[day**3*i*(i+1)*(1+day*K)**(-i-2) for i in range(1,NOP+1)]
    return np.sum(list)

#define h2prime(K), which is percentage allocation to options at different strike
def H2Prime(K,N,n,day):
    return (-IRR2prime(K,N,n,day)*K-2*IRR1prime(K,N,n,day))/IRR(K,N,n,day)**2 + \
            2*IRR1prime(K,N,n,day)**2*K/(IRR(K,N,n,day)**3)

#define function for CMS rate
def CMS(F,N,n,T,day,alpha, beta, rho, nu):    
    Rec_Integral = quad(lambda x:H2Prime(x,N,n,day)*Black76Rec(F, x, T, SABR(F, x, T, alpha, beta, rho, nu)),0,F)[0]
    Pay_Integral = quad(lambda x:H2Prime(x,N,n,day)*Black76Pay(F, x, T, SABR(F, x, T, alpha, beta, rho, nu)),F,1)[0]
    return (F + IRR(F,N,n,day)*(Rec_Integral + Pay_Integral))

# Solve Question 2 of Part3 
# Plug in CMS Calibration
Cal_Alpha = np.array([0.1390700710916347,0.1846448473766405,0.19685056373998003,0.17806011130792126,0.171099485983714,
                      0.1663491012283735,0.19855606733936929,0.20893906718727479,0.1860663780327828,0.18180077469733916,
                      0.17540856123443307,0.19402503686839012,0.19901184346119835,0.18669670221321796,0.17127061741482758])
Cal_Nu = np.array([2.049525533150364,1.6774866429530588,1.4382003710597762,1.0649646257191479,0.778196398296205,
                    1.338988878041516,1.0595787219736572,0.9361259029078985,0.6862515113342618,0.4654185152215292,
                    1.000043152897544,0.92292532993847,0.8521061837730378,0.7101097145693203,0.5950340162723781])
Cal_Rho = np.array([-0.6332193032745367,-0.5251242986310645,-0.4828554463440235,-0.4144646583009487,-0.2646156713921493,
                    -0.5842763238509328,-0.5423168017057935,-0.544466832860725,-0.47917055366840167,-0.48508466849870013,
                    -0.5384882907533526,-0.5407507527146742,-0.5280185142134435,-0.5117959528006836,-0.46248079391229163])    
    
CMSList=[]
Q2 = (Part_1.ps_df).copy()
for i in range(len(Q2)):
    n=Q2['fwd'][i]
    T=n
    N=Q2['swap'][i]+n
    F=Q2['par_swap_rate'][i]
    #D=Part_1.df['DF'].values[Part_1.df['Tenor'].values==T][0]
    alpha = Cal_Alpha[i]
    rho = Cal_Rho[i]
    nu = Cal_Nu[i]
    CMSList += [CMS(F,N,n,T, 1, alpha, 0.9, rho, nu)]
    
CMSList = pd.DataFrame(CMSList, index=Part_1.ps_df.index)
Q2['CMS Rate']=CMSList
Q2['CMS-Forward'] = (Q2['CMS Rate'] - Part_1.ps_df['par_swap_rate']).values
print(Q2)



# Solve Question 1 of Part3
# PV of CMS10Y
Inter_Alpha10 = np.zeros(10)
Inter_Alpha10[0] = Cal_Alpha[4]
Inter_Rho10 = np.zeros(10)
Inter_Rho10[0] = Cal_Rho[4]
Inter_Nu10 = np.zeros(10)
Inter_Nu10[0] = Cal_Nu[4]
Inter_CMS10Y = np.zeros(10)
Inter_CMS10Y[0] = Q2['par_swap_rate'][4]

Alpha10 = CubicSpline([1, 5, 10], [Cal_Alpha[4], Cal_Alpha[9], Cal_Alpha[14]])
Rho10 = CubicSpline([1, 5, 10], [Cal_Rho[4], Cal_Rho[9], Cal_Rho[14]])
Nu10 = CubicSpline([1, 5, 10], [Cal_Nu[4], Cal_Nu[9], Cal_Nu[14]])
x10 = np.linspace(1, 5, 9)

for i in range(len(x10)):
    Inter_Alpha10[i+1] = Alpha10(x10[i])
    Inter_Rho10[i+1] = Rho10(x10[i])
    Inter_Nu10[i+1] = Nu10(x10[i])
    Inter_CMS10Y[i+1] = Part_1.Par_Swap_Solver(x10[i],10,0.5)

Expiry10 = np.linspace(0.5, 5, 10)
DF = Part_1.df

CMS10List=[]

for i in range(len(Expiry10)):
    df10 = (DF['DF'].values[DF['Tenor'].values==Expiry10[i]])[0]
    n = Expiry10[i]
    T = n
    N = Expiry10[i]+10
    F = Inter_CMS10Y[i]
    alpha = Inter_Alpha10[i]
    rho = Inter_Rho10[i]
    nu = Inter_Nu10[i]
    0.5*df10*CMS(F,N,n,T,0.5,alpha, 0.9, rho, nu)
    CMS10List += [0.5*df10*CMS(F,N,n,T,0.5,alpha, 0.9, rho, nu)]

print("The present value of CMS10Y leg is %s" %(np.sum(CMS10List)))

    
# PV of CMS2Y
Inter_Alpha2 = np.zeros(40)
Inter_Alpha2[0:3] = Cal_Alpha[1]
Inter_Rho2 = np.zeros(40)
Inter_Rho2[0:3] = Cal_Rho[1]
Inter_Nu2 = np.zeros(40)
Inter_Nu2[0:3] = Cal_Nu[1]
Inter_CMS2Y = np.zeros(40)
Inter_CMS2Y[0:3] = Q2['par_swap_rate'][1]


Alpha2 = CubicSpline([1, 5, 10], [Cal_Alpha[1], Cal_Alpha[6], Cal_Alpha[11]])
Rho2 = CubicSpline([1, 5, 10], [Cal_Rho[1], Cal_Rho[6], Cal_Rho[11]])
Nu2 = CubicSpline([1, 5, 10], [Cal_Nu[1], Cal_Nu[6], Cal_Nu[11]])
x2 = np.linspace(1, 10, 37)

for i in range(len(x2)):
    Inter_Alpha2[i+3] = Alpha2(x2[i])
    Inter_Rho2[i+3] = Rho2(x2[i])
    Inter_Nu2[i+3] = Nu2(x2[i])    
#    Inter_CMS2Y[i+3] = Part_1.Par_Swap_Solver(x2[i],2,0.25)

Expiry2 = np.linspace(0.25, 10, 40)
DF = Part_1.df

#Interpolate DF and L_DF
DF2=pd.DataFrame(columns=['Tenor'])
DF2['Tenor']=np.linspace(0.25,12,48)

DF.set_index('Tenor',drop=False,inplace=True)
DF2.set_index('Tenor',drop=False,inplace=True)

DF3=pd.concat([DF2,DF],axis=1,sort=True)
#DF3.dropna(axis=1,how='all')
DF3.drop(columns=DF3.columns[0:2],inplace=True)

DF3.iloc[0,2]=DF3.iloc[1,2]
for i in range(1,24):
    DF3.iloc[i*2,2]=1/2*(DF3.iloc[i*2-1,2]+DF3.iloc[i*2+1,2])

DF3.iloc[0,5]=DF3.iloc[1,5]
for i in range(1,24):
    DF3.iloc[i*2,5]=1/2*(DF3.iloc[i*2-1,5]+DF3.iloc[i*2+1,5])

#Computation for CMS leg    
CMS2Leg=[]
Flist=[]
for i in range(len(Expiry2)):
    day=0.25
    n = Expiry2[i]
    T = n
    N = Expiry2[i]+2
    df2=DF3['DF'][n]
    Float = DF3['L_DF'][n]-DF3['L_DF'][N]
    Fix=[(day*DF3['L_DF'][n+m]) for m in np.linspace(0.25,2,8)]
    FixLeg=np.sum(Fix)
    F=Float/FixLeg
    Flist += [F]
    alpha = Inter_Alpha2[i]
    rho = Inter_Rho2[i]
    nu = Inter_Nu2[i]
#    0.25*df2*CMS(F,N,n,T,025,alpha, 0.9, rho, nu)
    CMS2Leg += [day*df2*CMS(F,N,n,T,0.25,alpha, 0.9, rho, nu)]

print("The present value of CMS2Y leg is %s" %(np.sum(CMS2Leg)))



