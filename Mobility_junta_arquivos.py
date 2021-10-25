# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:05:34 2021

@author: Adhimar
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy.constants as ct


n_runs = 40        # número de vezes que o programa irá rodar
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
    return dp_2deg(indv[0],indv[1],indv[3],indv[2],indv[9]),pz_2deg(indv[0], indv[1], indv[3], indv[2],indv[9],indv[5], indv[4] ), polar_2deg(indv[8],indv[7],indv[6], indv[3],indv[9]), alloy_2deg(indv[7], indv[8], indv[3]),back_imp(indv[7], indv[3], indv[10]),dislocation_2deg(indv[8], indv[3], indv[7],indv[11]), Matthiessen(np.array([dp_2deg(indv[0],indv[1],indv[3],indv[2],indv[9]), pz_2deg(indv[0], indv[1], indv[3], indv[2],indv[9],indv[5], indv[4] ),polar_2deg(indv[8],indv[7],indv[6], indv[3],indv[9]), alloy_2deg(indv[7], indv[8], indv[3]),back_imp(indv[7], indv[3], indv[10]), dislocation_2deg(indv[8], indv[3], indv[7],indv[11])]))

def graficomodelo(Temperature, Mobility,Resistivity, Carrier, dp, pz, polar, alloy, back,disloc, model,title):
    plt.clf()
    plt.plot(Temperature,  Mobility, 'k*', linewidth = 2,label='Data')
    plt.plot(Temperature,  model, 'k', linewidth = 2,label='Model')
    plt.plot(Temperature,  dp, 'cD', linewidth = 2,label='Deformation Potential')
    plt.plot(Temperature,  pz, 'ro', linewidth = 2,label='Piezoelectric')
    plt.plot(Temperature,  polar, 'bo', linewidth = 2,label='Polar optical')
    plt.plot(Temperature,  alloy, 'ms', linewidth = 2,label='Alloy')
    plt.plot(Temperature,  back, 'gD', linewidth = 2,label='Back Impurity')
    plt.plot(Temperature,  disloc, 'go', linewidth = 2,label='Dislocation')
    plt.xlabel('$T(K)$')
    plt.ylabel('$cm^2/Vs$')
    plt.legend()

    plt.savefig(name+'_'+title+'_model.png', dpi=300)
    
    plt.clf()
    plt.plot(Temperature,  Carrier, 'k*', linewidth = 2,label='Data')
    plt.xlabel('$T(K)$')
    plt.ylabel('$n(cm^{-2})$')
    plt.legend()

    plt.savefig(name+'_'+title+'_carrier.png', dpi=300)



out = np.zeros((n_runs, 13)) 
# rho, ul, ed,mef, h14, ut, epsylon_inf, epsylon, x ll, nb, ndis, tpo
for j in range(n_runs):
    data = np.loadtxt(str('results_')+str(name)+"_"+str(j)+".txt")
    out[j,:]  = data[gmax-1, 2:15]  

dp, pz, polar, alloy, back,disloc, model = simulatedd(out[0,:])

graficomodelo(Temperature, Mobility,Resistivity, Carrier, dp, pz, polar, alloy, back,disloc, model,'visualizar')

rho = out[:,0]
ul = out[:,1]
ed = out[:,2]
mef = out[:,3]
h14 = out[:,4]
ut = out[:,5]
epsylon_inf = out[:,6]
epsylon = out[:,7]
x = out[:,8]
ll = out[:,9]
nb = out[:,10]
ndis = out[:,11]
tpo1 = out[:,12]


b = open(str('results_')+str(name)+"_"+"all_values.txt", "w")
b.write('runs = ' + str(n_runs)+'\n')
b.write('popsize = ' + str(popsize)+'\n')
b.write('gmax = ' + str(gmax)+'\n')
b.write('rho = ' + str(np.mean(rho)) +'$\pm$'+ str(np.std(rho))+'\n')
b.write('ul = ' + str(np.mean(ul)) +'$\pm$'+ str(np.std(ul))+'\n')
b.write('ed = ' + str(np.mean(ed)) +'$\pm$'+ str(np.std(ed))+'\n')
b.write('mef = ' + str(np.mean(mef)) +'$\pm$'+ str(np.std(mef))+'\n')
b.write('h14 = ' + str(np.mean(h14)) +'$\pm$'+ str(np.std(h14))+'\n')
b.write('ut = ' + str(np.mean(ut)) +'$\pm$'+ str(np.std(ut))+'\n')
b.write('epsylon_inf = ' + str(np.mean(epsylon_inf)) +'$\pm$'+ str(np.std(epsylon_inf))+'\n')
b.write('epsylon = ' + str(np.mean(epsylon)) +'$\pm$'+ str(np.std(epsylon))+'\n')
b.write('x = ' + str(np.mean(x)) +'$\pm$'+ str(np.std(x))+'\n')
b.write('ll = ' + str(np.mean(ll)) +'$\pm$'+ str(np.std(ll))+'\n')
b.write('nb = ' + str(np.mean(nb)) +'$\pm$'+ str(np.std(nb))+'\n')
b.write('ndis = ' + str(np.mean(ndis)) +'$\pm$'+ str(np.std(ndis))+'\n')
b.write('tpo = ' + str(np.mean(tpo1)) +'$\pm$'+ str(np.std(tpo1))+'\n')
b.close()

 
# Salva dados no arquivo
x = np.transpose(np.array([Temperature, Mobility,Resistivity, Carrier, dp, pz, polar, alloy, back,disloc, model]))
np.savetxt(name+'_out_to_plot_final.dat', x, fmt='%1.4e',delimiter=' ', header='Temperature Mobility Resistivity Carrier dp pz polar alloy back disloc model')