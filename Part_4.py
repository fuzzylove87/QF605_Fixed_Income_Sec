# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:41:46 2019

@author: Woon Tian Yong
"""

import numpy as np
import pandas as pd
import Part_1
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.integrate import quad

def IRR(S,n,N,Delta):
    IRR=0
    for i in range(N-n):
        IRR+=(Delta)/((1+Delta*S)**(i+1))
    return IRR

def IRR_d1(S,n,N,Delta):
    IRR=0
    for i in range(N-n):
        IRR+=(Delta**2*(-(i+1)))/((1+Delta*S)**(i+2))
    return IRR

def IRR_d2(S,n,N,Delta):
    IRR=0
    for i in range(N-n):
        IRR+=(Delta**3*(i+2)*(i+1))/((1+Delta*S)**(i+3))
    return IRR

def h_d2(S,n,N,Delta):
    return ((-IRR_d2(S,n,N,Delta)*S-2*IRR_d1(S,n,N,Delta))/(IRR(S,n,N,Delta)**2)
        +(2*(IRR_d1(S,n,N,Delta)**2)*S)/(IRR(S,n,N,Delta)**3))

def g_P4(S):
    return (S**(1/4))-0.2

def g_d1_P4(S):
    return (1/4)*(S**(-3/4))

def g_d2_P4(S):
    return (-3/16)*(S**(-7/4))

def h_P4(S,n,N,Delta):
    return g_P4(S)/IRR(S,n,N,Delta)

def h_d1_P4(S,n,N,Delta):
    return (IRR(S,n,N,Delta)*g_d1_P4(S)-g_P4(S)*IRR_d1(S,n,N,Delta))/(IRR(S,n,N,Delta)**2)

def h_d2_P4(S,n,N,Delta):
    return ((IRR(S,n,N,Delta)*g_d2_P4(S)
            -IRR_d2(S,n,N,Delta)*g_P4(S)
            -2*IRR_d1(S,n,N,Delta)*g_d1_P4(S))/
            (IRR(S,n,N,Delta)**2)
            +(2*(IRR_d1(S,n,N,Delta)**2)*g_P4(S))/
            (IRR(S,n,N,Delta)**3))
    
def B76_LogN_Pay(S,n,N,DF,Delta,K,T,sigma):
    x_star = (np.log(K/S)+(sigma**2)*T/2)/(sigma*np.sqrt(T))
    return DF*IRR(S,n,N,Delta)*(S*ss.norm.cdf(-x_star+sigma*np.sqrt(T)) \
            -K*ss.norm.cdf(-x_star))

def B76_LogN_Rec(S,n,N,DF,Delta,K,T,sigma):
    x_star = (np.log(K/S)+(sigma**2)*T/2)/(sigma*np.sqrt(T))
    return DF*IRR(S,n,N,Delta)*(K*ss.norm.cdf(x_star) \
            -S*ss.norm.cdf(x_star-sigma*np.sqrt(T)))
    
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

def CMS(S,n,N,DF,Delta,K,T,alpha,beta,rho,nu,upper):
    g=S
    integral_pay=quad(lambda x: h_d2(x,n,N,Delta)*B76_LogN_Pay(S,n,N,DF,Delta,x,T,SABR(S, x, T, alpha, beta, rho, nu)),S,upper)
    integral_rec=quad(lambda x: h_d2(x,n,N,Delta)*B76_LogN_Rec(S,n,N,DF,Delta,x,T,SABR(S, x, T, alpha, beta, rho, nu)),0,S)
    return (g+(1/DF)*(integral_pay[0]+integral_rec[0]))

def CMS_P4(S,n,N,DF,Delta,K,T,alpha,beta,rho,nu,upper):
    g=g_P4(S)
    integral_pay=quad(lambda x: h_d2_P4(x,n,N,Delta)*B76_LogN_Pay(S,n,N,DF,Delta,x,T,SABR(S, x, T, alpha, beta, rho, nu)),S,upper)
    integral_rec=quad(lambda x: h_d2_P4(x,n,N,Delta)*B76_LogN_Rec(S,n,N,DF,Delta,x,T,SABR(S, x, T, alpha, beta, rho, nu)),0,S)
    return g*DF+integral_pay[0]+integral_rec[0] 

def CMS_Caplet_P4(S,n,N,DF,Delta,K,T,L,alpha,beta,rho,nu,upper):
    integral_pay=quad(lambda x: h_d2_P4(x,n,N,Delta)*B76_LogN_Pay(S,n,N,DF,Delta,x,T,SABR(S, x, T, alpha, beta, rho, nu)),L,upper)
    return (h_d1_P4(L,n,N,Delta)*
            B76_LogN_Pay(S,n,N,DF,Delta,L,T,SABR(S, L, T, alpha, beta, rho, nu))
            +integral_pay[0])

'''
ps_df=pd.read_csv('ps_df.csv',index_col=0)
IR_df=pd.read_csv('Part 1 Data.csv',index_col=0)
'''

SABR_df=pd.read_csv('SABR.csv')

Start=[1,5,10]
Tenor=[1,2,3,5,10]

CMS_df=pd.DataFrame(np.zeros((len(Start)*len(Tenor),5)),columns=['Start','Tenor','CMS Rate','CMS Rate P4','CMS Caplet P4'])

Delta=1

### For CMSes ###

for i in range(len(Start)):
    for j in range(len(Tenor)):
        loc=(i)*len(Tenor)+j
        CMS_df.iloc[loc,0]=Start[i]
        CMS_df.iloc[loc,1]=Tenor[j]
        
        n=T=Start[i]
        N=Start[i]+Tenor[j]
        
        DF=Part_1.df['DF'][Start[i]*2-1]
        
        S=Part_1.ps_df.iloc[loc,2]
        
        alpha=SABR_df['Alpha'][loc]
        beta=SABR_df['Beta'][loc]
        rho=SABR_df['Rho'][loc]
        nu=SABR_df['Nu'][loc]
        
        K=S
        upper=0.85
        
        CMS_df.iloc[loc,2]=CMS(S,n,N,DF,Delta,K,T,alpha,beta,rho,nu,upper)
        CMS_df.iloc[loc,3]=CMS_P4(S,n,N,DF,Delta,K,T,alpha,beta,rho,nu,upper)
        
        L=0.0016
        
        CMS_df.iloc[loc,4]=CMS_Caplet_P4(S,n,N,DF,Delta,K,T,L,alpha,beta,rho,nu,upper)
    

'''
plt.figure(figsize=(9,4.5))
plt.scatter(Part_1.df['Tenor'],Part_1.df['OIS'],color='b',label='OIS',zorder=3)
plt.xlabel('Tenor',fontweight='bold')
plt.ylabel('Interest Rate',fontweight='bold')
plt.grid(zorder=0)
plt.legend(loc='upper right')
plt.axis([0,35,0.002,0.006])
plt.show()

plt.figure(figsize=(9,4.5))
plt.scatter(Part_1.df['Tenor'],Part_1.df['L'],color='r',label='LIBOR',zorder=3)
plt.xlabel('Tenor',fontweight='bold')
plt.ylabel('Interest Rate',fontweight='bold')
plt.grid(zorder=0)
plt.legend(loc='upper right')
plt.axis([0,35,0.02,0.06])
plt.show()
'''

plt.figure(figsize=(6,4.5))
plt.plot(CMS_df['Tenor'][0:5],CMS_df['CMS Rate'][0:5]**(1/4),color='k',linewidth=2.5,label='Spot Price of Underlying',zorder=3)
plt.scatter(CMS_df['Tenor'][0:5],CMS_df['CMS Rate'][0:5]**(1/4),zorder=4,edgecolors='k',label='_nolegend_',linewidths=2.5,c='w',marker='o',s=35)
plt.plot(CMS_df['Tenor'][0:5],CMS_df['CMS Rate P4'][0:5],color='b',linewidth=2.5,label='CMS Forward Contract Value',zorder=3)
plt.scatter(CMS_df['Tenor'][0:5],CMS_df['CMS Rate P4'][0:5],zorder=4,edgecolors='b',label='_nolegend_',linewidths=2.5,c='w',marker='o',s=35)
plt.plot(CMS_df['Tenor'][0:5],CMS_df['CMS Caplet P4'][0:5],color='r',linewidth=2.5,label='CMS Caplet Value',zorder=3)
plt.scatter(CMS_df['Tenor'][0:5],CMS_df['CMS Caplet P4'][0:5],zorder=4,edgecolors='r',label='_nolegend_',linewidths=2.5,c='w',marker='o',s=35)
plt.xlabel('Tenor (N-n)', fontweight='bold')
plt.ylabel('Contract Value', fontweight='bold')
plt.title('When n=1', fontweight='bold')
plt.grid(zorder=0)
plt.legend(loc='lower right')
plt.axis([0,11,0.14,0.47])
plt.savefig('P4n1',dpi=800)
plt.show()

plt.figure(figsize=(6,4.5))
plt.plot(CMS_df['Tenor'][5:10],CMS_df['CMS Rate'][5:10]**(1/4),color='k',linewidth=2.5,label='Spot Price of Underlying',zorder=3)
plt.scatter(CMS_df['Tenor'][5:10],CMS_df['CMS Rate'][5:10]**(1/4),zorder=4,edgecolors='k',label='_nolegend_',linewidths=2.5,c='w',marker='o',s=35)
plt.plot(CMS_df['Tenor'][5:10],CMS_df['CMS Rate P4'][5:10],color='b',linewidth=2.5,label='CMS Forward Contract Value',zorder=3)
plt.scatter(CMS_df['Tenor'][5:10],CMS_df['CMS Rate P4'][5:10],zorder=4,edgecolors='b',label='_nolegend_',linewidths=2.5,c='w',marker='o',s=35)
plt.plot(CMS_df['Tenor'][5:10],CMS_df['CMS Caplet P4'][5:10],color='r',linewidth=2.5,label='CMS Caplet Value',zorder=3)
plt.scatter(CMS_df['Tenor'][5:10],CMS_df['CMS Caplet P4'][5:10],zorder=4,edgecolors='r',label='_nolegend_',linewidths=2.5,c='w',marker='o',s=35)
plt.xlabel('Tenor (N-n)', fontweight='bold')
plt.ylabel('Contract Value', fontweight='bold')
plt.title('When n=5', fontweight='bold')
plt.grid(zorder=0)
plt.legend(loc='lower right')
plt.axis([0,11,0,0.54])
plt.savefig('P4n5',dpi=800)
plt.show()

plt.figure(figsize=(6,4.5))
plt.plot(CMS_df['Tenor'][10:15],CMS_df['CMS Rate'][10:15]**(1/4),color='k',linewidth=2.5,label='Spot Price of Underlying',zorder=3)
plt.scatter(CMS_df['Tenor'][10:15],CMS_df['CMS Rate'][10:15]**(1/4),zorder=4,edgecolors='k',label='_nolegend_',linewidths=2.5,c='w',marker='o',s=35)
plt.plot(CMS_df['Tenor'][10:15],CMS_df['CMS Rate P4'][10:15],color='b',linewidth=2.5,label='CMS Forward Contract Value',zorder=3)
plt.scatter(CMS_df['Tenor'][10:15],CMS_df['CMS Rate P4'][10:15],zorder=4,edgecolors='b',label='_nolegend_',linewidths=2.5,c='w',marker='o',s=35)
plt.plot(CMS_df['Tenor'][10:15],CMS_df['CMS Caplet P4'][10:15],color='r',linewidth=2.5,label='CMS Caplet Value',zorder=3)
plt.scatter(CMS_df['Tenor'][10:15],CMS_df['CMS Caplet P4'][10:15],zorder=4,edgecolors='r',label='_nolegend_',linewidths=2.5,c='w',marker='o',s=35)
plt.xlabel('Tenor (N-n)', fontweight='bold')
plt.ylabel('Contract Value', fontweight='bold')
plt.title('When n=10', fontweight='bold')
plt.grid(zorder=0)
plt.legend(loc='lower right')
plt.axis([0,11,-0.1,0.65])
plt.savefig('P4n10',dpi=800)
plt.show()




    