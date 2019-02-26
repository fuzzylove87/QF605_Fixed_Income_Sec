
# coding: utf-8

# In[239]:


import pandas as pd
import numpy as np
import scipy.stats as ss
from scipy.optimize import brentq
import matplotlib.pylab as plt
import datetime as dt
from scipy import interpolate
from scipy.optimize import least_squares
from math import exp


# Import data

# In[2]:


data=pd.read_csv('IR_Data.csv',usecols=range(13),index_col=['Expiry','Tenor'],skiprows=[16,17,18])
data.head()


# In[3]:


Swap=pd.read_excel('ATM.xlsx')
Swap


# In[178]:


Expiry=np.array(['1Y','5Y','10Y'])
Tenor=np.array(['1Y','2Y','3Y','5Y','10Y'])
s=np.array([-200,-150,-100,-50,-25,0,25,50,100,150,200])
Notion=10_000


# SABR Calibration

# In[83]:


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


# In[84]:


def sabrcalibration(x, strikes, vols, F, T):
    err = 0.0
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T,
                           x[0], 0.9, x[1], x[2]))**2
    return err


# In[171]:


SABR_Alpha=pd.DataFrame(np.zeros((3,5)),index=Expiry,columns=Tenor)
SABR_Rho=pd.DataFrame(np.zeros((3,5)),index=Expiry,columns=Tenor)
SABR_Nu=pd.DataFrame(np.zeros((3,5)),index=Expiry,columns=Tenor)
initialGuess_sabr=[0.1,-1,1]


# In[172]:


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


# In[175]:


SABR_Alpha


# In[176]:


SABR_Rho


# In[177]:


SABR_Nu


# In[154]:


for ex in Expiry:
    for te,i in zip(Tenor,range(6)):
        print(ex,te)
        ATM=Swap[te][ex] #ATM and K changes
        strikes=s*0.0001+ATM
        T=int(ex[:-1])
        #plt.subplot(3,2,i+1)
        plt.plot(strikes,data.loc[ex,te].values/100,'go')
        plt.plot(strikes,[SABR(ATM,x,T,SABR_Alpha.loc[ex,te],0.9,SABR_Rho.loc[ex,te],SABR_Nu.loc[ex,te]) for x in strikes])
        plt.show()


# Displaced-Diffusion Calibration

# In[99]:


def DD_Model (F, K, sigma, T, Beta, TYPE):
    r=1
    if Beta==0:
        x_Star = (K-F)/(F*sigma*np.sqrt(T))
        Call=np.exp(-r*T)*((F-K)*ss.norm.cdf(-x_Star,0,1)+
                           F*sigma*np.sqrt(T)*ss.norm.pdf(-x_Star,0,1))
        Put=np.exp(-r*T)*((K-F)*ss.norm.cdf(x_Star,0,1)+
                             F*sigma*np.sqrt(T)*ss.norm.pdf(-x_Star,0,1))
    else:
        F=F/Beta
        K=K+((1-Beta))*F
        sigma=Beta*sigma
        d1 = (np.log(F/K)+(sigma**2)/2*T)/(sigma*np.sqrt(T))
        d2 = (np.log(F/K)-(sigma**2)/2*T)/(sigma*np.sqrt(T))
        Call = np.exp(-r*T)*(F*ss.norm.cdf(d1,0,1)-K*ss.norm.cdf(d2,0,1))
        Put = np.exp(-r*T)*(K*ss.norm.cdf(-d2,0,1)-F*ss.norm.cdf(-d1,0,1))
    if TYPE == 'Put':
        return Put
    else:
        return Call


# In[282]:


def impliedVolatility(S, K, price, T, Beta):
    impliedVol = brentq(lambda x: price - 
                        DD_Model(S, K,  x, T, 1, 'Call'),
                        1e-6, 1)
    return impliedVol


# In[247]:


def DD_Calibrate_Price(x,ATM,strikes,T,vols):
    err=0.0
    for i,vol in zip(range(len(strikes)), vols):
        price1 = Notion*DD_Model(ATM, strikes[i], vol, T, 1,'Call') #Beta=1 using Black76
        price2 = Notion*DD_Model(ATM, strikes[i], x[0], T, x[1], 'Call')
        err += (price1 - price2)**2
        #plt.scatter(strikes[i],price1,color='green')
    return err


# In[272]:


initialGuess_DD=[0.5,0.2]
Cal_Beta=pd.DataFrame(np.zeros((3,5)),index=Expiry,columns=Tenor)
Cal_Sigma=pd.DataFrame(np.zeros((3,5)),index=Expiry,columns=Tenor)


# In[273]:


for ex in Expiry:
    for te in Tenor:             #Set sigma,T,ATM,strikes
        sigma=data.loc[(ex,te), 'ATM']/100
        T=int(ex[:-1])
        ATM=Swap[te][ex] #ATM and K changes
        strikes=s*0.0001+ATM
        res = least_squares(lambda x: DD_Calibrate_Price(x,
                                          ATM,
                                          strikes,
                                          T,                                          
                                          data.loc[ex,te].values/100),
                                          initialGuess_DD,
                                          bounds=([0,0],[np.inf,1]))
        Cal_Beta[te][ex]= res.x[1]
        Cal_Sigma[te][ex]=res.x[0] 


# In[274]:


Cal_Beta


# In[275]:


Cal_Sigma


# In[293]:


imp=[impliedVolatility(ATM, x, p, T, 0.218611) for x,p in zip(strikes,price_DD)]
imp


# In[292]:


ATM=0.038436
sigma=0.2447
T=1
s=np.array([-200,-150,-100,-50,-25,0,25,50,100,150,200])
strikes=s*0.0001+ATM
price_DD=[DD_Model(ATM,x, data.loc['1Y','10Y'][i]/100, T,0.218611 ,'Call') for i,x in zip(range(len(strikes)),strikes)]
price_DD


# In[294]:


plt.plot(strikes,data.loc['1Y','10Y']/100,'go')
plt.plot(strikes,imp)


# In[259]:


initialGuess_DD=[0.1,0.2]
ATM=0.038436
sigma=0.2447
T=1
s=np.array([-200,-150,-100,-50,-25,0,25,50,100,150,200])
strikes=s*0.0001+ATM
strikes


# In[260]:


res = least_squares(lambda x: DD_Calibrate_Price(x,
                                          ATM,
                                          strikes,
                                          T,                
                                          data.loc['1Y','10Y'].values/100),
                                          initialGuess_DD,
                                          bounds=([0,0],[np.inf,1]))


# In[261]:


sigma=res.x[0]
beta=res.x[1]


# In[262]:


sigma


# In[263]:


beta


# In[126]:


plt.plot(strikes,[Notion*DD_Model(ATM,x, data.loc['10Y','10Y'][i]/100, T, 1,'Call') for i,x in zip(range(len(strikes)),strikes)],'go')
plt.plot(strikes,[Notion*DD_Model(ATM, x, sigma, T,0.295704311, 'Call') for x in strikes])


# In[116]:


data.loc['10Y','10Y'][1]


# In[108]:


[Notion*DD_Model(ATM, x, sigma, T,0.295704311, 'Call') for x in strikes]


# PVBP Calculation

# In[189]:


df=pd.read_csv('E:/Github/QF605_Fixed_Income_Sec/df.csv')
df.head()


# In[190]:


def PVBP(start,stop,coupon_period=0.5):
    value=0.0
    time=np.arange(start,stop+0.1,0.5)
    for i in time:
        value=value+df[df['Tenor']==i].DF.values
    return value*coupon_period


# Q3: Pricing the following swaption

#  payer 2y x 10y K = 1%; 2%; 3%; 4%; 5%; 6%; 7%; 8%

# In[226]:


from scipy.interpolate import interp1d


# In[228]:


def interpol(x1,y1,x2,y2,x):
    return y1+x*((y2-y1)/(x2-x1))


# In[232]:


strike=np.arange(0.01,0.081,0.01)
T=2
ATM=interpol(1,26.355,5,22.250,2)
ATM


# In[237]:


alpha=interpol(1,0.156374,5,0.181581,2)
rho=interpol(1,-0.193758,5,-0.070987,2)
nu=interpol(1,0.534328,5,0.205102,2)


# In[240]:


def Black76Pay(F, K, T, sigma):
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return (F*ss.norm.cdf(d1) - K*ss.norm.cdf(d2))

def Black76Rec(F, K, T, sigma):
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return (K*ss.norm.cdf(-d2) - F*ss.norm.cdf(-d1))


# In[245]:


value=[Notion*Black76Pay(ATM,x,T,SABR(ATM,x,T,alpha,0.9,rho,nu)) for x in strike]


# In[246]:


value

