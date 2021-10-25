# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:05:34 2021

@author: Adhimar
"""

import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
from multiprocessing import Pool
# from uncertainties import ufloat
# from uncertainties.umath import *
import scipy.constants as ct
import time

starttime = time.time()

n_runs = 40      # número de vezes que o programa irá rodar
popsize = 100      # número de possíveis soluções
gmax = 50000   # número de iterações

#constantes
epsylon_0 = ct.epsilon_0
m         = ct.m_e
k         = ct.k
h_bar     = ct.hbar
e         = ct.e
lc        = 6.0583e-10  #Lattice constant InAs 
a1 = 6.0583e-10  #InAs
a0 = 5.65325e-10 #GaAs


name = "BH9824"
data = np.genfromtxt(name+".dat", names=True)
Temperature = data['T']    #K
Mobility    = data['m']    #cm2/Vs
Resistivity = data['rho']  #2D
Carrier     = data['n']    #cm-2

mef =[0.02,0.4]
rho = [5.0, 5.7]
ul =[4.5e5,5.2e5]
ut= [4.2e5,5.2e5]
ed= [6.0,10.0]
h14 =[1.0e7,1.0e9]
epsilon_inf=[10.8,12.5]
epsilon_s = [12.9,15.3]
x = [0.4,0.6]
z0 = [1.0,4.0]
nic = [1.e7,1.e11]
ndis = [1.e7,1.e10]

# rho, ul, ed,mef, h14, ut, epsylon_inf, epsylon, x ll, nb, ndis
lbound      = np.array([rho[0], ul[0], ed[0], mef[0], h14[0], ut[0], epsilon_inf[0], epsilon_s[0], x[0], z0[0], nic[0], ndis[0],0.0, 0.0]) 
ubound      = np.array([rho[1], ul[1], ed[1], mef[1], h14[1], ut[1], epsilon_inf[1], epsilon_s[1], x[1], z0[1], nic[1], ndis[1], 2.0, 1.0])

def n_m2_to_cm2(n):  # n de m-2 to cm-2
    return(n*1e-4)
def n_cm2_to_m2(n):  # n de cm-2 to m-2
    return(n*1e4)
def MU_cm2_to_m2(n): #mobilidade cm2/Vs to m2/Vs 
    return(n*1e-4)
def MU_m2_to_cm2(n): #mobilidade m2/Vs to cm2/Vs  
    return(n*1e4)
def k_med(m_e):  
    return (2.*m_e*k*Temperature/(h_bar**2))**0.5
def dp_2deg(rho,ul,m_e,ed,l): #deformation potential acoustic phonon scattering
    ul  = ul* 1.e-2  #m/s
    rho = rho*1.e3   #kg/m^3
    ed  = ed*e       #J
    l   = l*lc
    m_e = m_e*m      #kg
    return MU_m2_to_cm2((e*(h_bar**3)*rho*(ul**2)*l)/((m_e**2)*(ed**2)*k*Temperature))
def pz_2deg(rho, ul, m_e, ed,l,ut, h14 ): #Piezoelectric-phonon scattering
    ul  = ul* 1.e-2  #m/s
    rho = rho*1.e3   #kg/m^3
    ed  = ed*e       #J
    l   = l*lc
    m_e = m_e*m      #kg
    ut  = ut* 1.e-2  #m/s
    gamma_t = (2.*h_bar*ut*k_med(m_e)/(k*Temperature))
    gamma_l = (2.*h_bar*ul*k_med(m_e)/(k*Temperature))
    I_gamma_t = ((((4.*gamma_t)/(3.*np.pi))**2 )+1.)**0.5
    I_gamma_l = ((((4.*gamma_l)/(3.*np.pi))**2 )+1.)**0.5
    aux = (((9./32.)+(13./32.)*((ul/ut)**2)*(I_gamma_t/I_gamma_l))**(-1.))
    m_dpp = (e*(h_bar**3)*rho*(ul**2)*l)/((m_e**2)*(ed**2)*k*Temperature)
    return MU_m2_to_cm2(((np.pi*k_med(m_e)*(ed**2))/(l*(e**2)*(h14**2)))*aux*m_dpp)

def polar_2deg(x, epsylon,epsylon_inf, m_e, l): #polar optical phonon scattering
    m_e = m_e*m      #kg
    l   = l*lc
    ep = 1./((1./epsylon_inf)-(1./epsylon))
    return MU_m2_to_cm2(((4.*np.pi*epsylon_0*ep*(h_bar**3))/(e*tpo(x)*k*(m_e**2)*l))*(np.exp(tpo(x)/Temperature)-1.))
def bbb(m_e,epsylon):
    return ((33.*(e**2)*m_e*n_cm2_to_m2(Carrier))/(8.*epsylon*(h_bar**2)))**0.3333333333333333
def Eg_InAs():
    return 0.415-(2.76e-4*(Temperature**2))/(Temperature+83.)
def alloy_2deg(epsylon, x, m_e):
    m_e = m_e*m      #kg
    epsylon = epsylon*epsylon_0
    b = bbb(m_e,epsylon)
    u_al = Eg_InAs()*e
    return MU_m2_to_cm2((16./(3.*b))*((e*(h_bar**3))/(x*(1.-x)*(m_e**2)*(3.48e-29)*(u_al**2)))) 
def so(epsylon):
    return (e**2)*n_cm2_to_m2(Carrier)/(2.*epsylon*k*Temperature)
def ib(m_e, epsylon):
    x = (so(epsylon)/(2.*k_med(m_e)))
    return (2.043*np.exp(-x/0.323)+0.654*np.exp(-x/1.669)+0.0063)
def back_imp(epsylon, m_e, nb):
    nb = n_cm2_to_m2(nb)  #m^-2
    m_e = m_e*m      #kg
    kf = k_med(m_e)
    epsylon = epsylon*epsylon_0
    return MU_m2_to_cm2((8.*np.pi*(h_bar**3)*(epsylon**2)*(kf**2))/((e**3)*(m_e**2)*nb*ib(m_e, epsylon)))
def parameter_xi(epsylon, m_e):
    ab = (4.*np.pi*epsylon*(h_bar**2))/((e**2)*m_e) # bohr radious
    qtf = 2./ab   #the 2D Thomas-Fermi wave vector
    return (2.*k_med(m_e)/qtf)
def it(epsylon, m_e):
    xi = parameter_xi(epsylon,m_e)
    return (-0.0275+0.2006*xi+0.4964*(xi**2)-0.1203*(xi**3))
def dislocation_2deg(x, m_e, epsylon,n_dis):
    n_dis = n_cm2_to_m2(n_dis)
    epsylon = epsylon*epsylon_0
    m_e = m_e*m
    k_f= k_med(m_e)
    c = x*a1 + (1.-x)*a0
    return MU_m2_to_cm2((4.*np.pi*(epsylon**2)*(h_bar**3)*(k_f**4)*(c**2))/((e**3)*(m_e**2)*n_dis*it(epsylon, m_e)))
def Matthiessen(vetor_mobility):
    soma = 0
    for i in range(0,len(vetor_mobility)):
        soma+= 1./vetor_mobility[i]
    return 1./soma

def tpo(x):
    return 420.-107.*x

def simulatedd(indv):
    return Matthiessen(np.array([dp_2deg(indv[0],indv[1],indv[3],indv[2],indv[9]), 
                                      pz_2deg(indv[0], indv[1], indv[3], indv[2],indv[9],indv[5], indv[4] ),
                                      polar_2deg(indv[8],indv[7],indv[6], indv[3],indv[9]),
                                      alloy_2deg(indv[7], indv[8], indv[3]),
                                      back_imp(indv[7], indv[3], indv[10]),
                                      dislocation_2deg(indv[8], indv[3], indv[7],indv[11])]))

def objective(indv):
    simulated = simulatedd(indv)
    return np.power((1/np.float(np.size(Mobility)))*np.sum(np.power(Mobility-simulated,2,dtype=np.float64),dtype=np.float64),0.5,dtype=np.float64)

def Pop():
    return lbound + (ubound-lbound)*np.random.rand(popsize, len(lbound))
def Donor(X, F, Xbest, Xr1, Xr2):
    return X + F * (Xbest - X) + F * (Xr1 - Xr2)
def Crossover(x, donor, sol):
    for j in range(len(lbound)):
        random_number = np.random.rand()
        if random_number <= x[-1] or j == np.random.choice(len(lbound)):
            sol[j] = donor[j]
        elif random_number > x[j]:
            sol[j] = x[j]
    return sol
def Penalty(sol):
    for j in range(len(lbound)):
        if sol[j] > ubound[j]:
            sol[j] = lbound[j] + np.random.rand()*(ubound[j]-lbound[j])
        if sol[j] < lbound[j]:
            sol[j] = lbound[j] + np.random.rand()*(ubound[j]-lbound[j])
    return sol
def Selection(sol, x):
    if objective(sol) < objective(x):
        x = sol
    else:
        x = x
    return x
def control(pop):
    if np.random.rand() < 0.1:
        pop[-2] = 0.1 + np.random.rand()*0.9 
    if np.random.rand() < 0.1:    
        pop[-1] = np.random.rand() 
    return pop
def main(count):
    a = open(str('results_')+str(name)+"_"+str(count)+".txt", "w")
    np.random.seed()
    g = 0
    pop = Pop()    # Gera a população inicial.
    print(pop)
    score = np.zeros((popsize, 1))   
    donor = np.zeros((popsize, len(lbound)))
    trial = np.zeros((popsize, len(lbound)))
    for i in range(popsize):
        score[i] = objective(pop[i,:])
    best_index = np.argmin(score)
    g+=1
    plt.ion()
    while g <= gmax:
        for i in range(popsize):
            random_index = np.random.choice(np.delete(np.arange(popsize),i),2)
            donor[i,:] = pop[i,:] + pop[i,-2]*(pop[best_index,:]-pop[i,:])+pop[i,-2]*(pop[random_index[0],:]-pop[random_index[1],:])
            trial[i,:] = Crossover(pop[i,:], donor[i,:], trial[i,:])
            trial[i,:] = Penalty(trial[i,:])
            pop[i,:] = Selection(trial[i,:], pop[i,:])
            pop[i,:] = control(pop[i,:])
            score[i] = objective(pop[i,:])
        best_index = np.argmin(score)
        a.write("%d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f \n" % (g, score[best_index], pop[best_index,0], pop[best_index,1], pop[best_index,2], pop[best_index,3], pop[best_index,4], pop[best_index,5], pop[best_index,6], pop[best_index,7], pop[best_index,8], pop[best_index,9], pop[best_index,10], pop[best_index,11]))   
        g+=1
    a.close()
    return 0
if __name__ == '__main__': 
    print("Processadores que serão usados = %d " % int(multiprocessing.cpu_count()))
    filenames = [i for i in range(n_runs)]
    p = Pool(int(multiprocessing.cpu_count()))
    p.map(main, filenames)
    
endtime = time.time()
tempo = (endtime - starttime)
print(tempo)