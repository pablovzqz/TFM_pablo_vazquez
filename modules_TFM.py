# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:58:08 2024

@author: pablo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.stats import binned_statistic_2d
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def profiles(xdata,ydata, threshold, n_bins=50 ,x_label=None, y_label=None, plot=False, normalize=False):
    '''
    xdata: np.array or 1 column dataframe with data for the x-axes
    ydata: np.array or 1 column dataframe with data for the x-axes
    n_bins: must be an integer, number of bins for the histogram. Default value: 50 could change if you had poor statistic
    threshold: lower limit for x_data (based on reliability of the lowest data)
    x_label: string that contains the name of the magnitude in the x axes
    y_label: string that contains the name of the magnitude in the y axes
    ######
    Output: returns a plot with the profile of ydata as a function of xdata. This profile is superposed over an histogram of the number of events so we can check if the statistic is good enough to consider one point as relevant.
    '''        
    ydata_filtered = np.array([s for s, d in zip(ydata, xdata) if d > threshold])
    xdata_filtered = np.array([i for i in xdata if i> threshold])
    
    mean_y, bin_edges, _ = binned_statistic(xdata_filtered, ydata_filtered, statistic='mean', bins=n_bins)
    std_y, _, _ = binned_statistic(xdata_filtered, ydata_filtered, statistic='std', bins=n_bins)
    count, _, _ = binned_statistic(xdata_filtered, ydata_filtered, statistic='count', bins=n_bins)


    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    if normalize==True:
        uncertainty = std_y/np.sqrt(count)
    else:
        uncertainty = std_y
    if plot==True:
        fig, ax1 = plt.subplots()
    
        ax1.plot(bin_centers, mean_y, 'bo', label='Average of {y_label}')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(f"Average {y_label}", color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title(f"Average {y_label} Profile vs {x_label}")
    
        ax2 = ax1.twinx()
        ax2.bar(bin_centers, count, width=(bin_edges[1] - bin_edges[0]), alpha=0.5, color='gray', label='Number of events', edgecolor='black')
        ax2.set_ylabel("Number of events", color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
    
        fig.tight_layout()
    
        plt.figure()
        plt.plot(bin_centers, uncertainty, 'b.')
        plt.ylabel(f'Uncertainty in {y_label}')
        plt.xlabel(f'{x_label}')
        plt.grid(True)
        
        plt.show()
    
    return bin_centers, mean_y, uncertainty
    
def filter_center_axial(Energy,Radial_position,DT,cut):
    '''
    

    Parameters
    ----------
    Energy : np.array with the energy of a set of events.
    Radial_position : radial position where those events took place.
    DT : axial position of the event.
    cut : maximum radio.

    Returns
    -------
    central_events_energy: list with the energy of the events that took place inside the cut.
    axial_position_central_events: list with the DT of the events inside the cut.
    external_events_energy: list with the energy of the events that took place outside the cut.
    axial_position_external_events: list with the DT of the events outside the cut.
    '''

    central_events_energy=[]
    axial_position_central_events=[]
    
    external_events_energy=[]
    axial_position_external_events=[]
  
    for i in range(len(Energy)):
        if DT[i]>0:
            if Radial_position[i]<=cut:
                central_events_energy.append(Energy[i])
                axial_position_central_events.append(DT[i])
            else:
                external_events_energy.append(Energy[i])
                axial_position_external_events.append(DT[i])
        
    print(f'The percentage of events insode the cut of {cut}mm is {np.round(len(axial_position_central_events)/len(DT)*100,3)}%')
    print(f'Then , the percentage outside the cut must be {np.round(len(axial_position_external_events)/len(DT)*100,3)}%')
    
    return central_events_energy,  axial_position_central_events, external_events_energy, axial_position_external_events

def interpolation_y(x_data, y_data, point, kind='linear'):
    '''
    x_data: vector with the centers of the bins in the histogram
    y_data: vector with the points of the profile
    point: value of the Y magnitud for the one I want to get X
    kind: type of interpolation, if not specified is linear, but it can be quadratic or cubic
    #####
    Output:
    value: value of the X
    '''
    interpolation_function=interp1d(y_data, x_data, kind, fill_value="extrapolate")
    value=interpolation_function(point)
    
    return value

def interpolation(x_data, y_data, point, kind='linear'):
    '''
    x_data: vector with the centers of the bins in the histogram
    y_data: vector with the points of the profile
    point: value of the Y magnitud for the one I want to get X
    kind: type of interpolation, if not specified is linear, but it can be quadratic or cubic
    #####
    Output:
    value: value of the X
    '''
    interpolation_function=interp1d(x_data, y_data, kind, fill_value="extrapolate")
    value=interpolation_function(point)
    return value


def predictions_y(x_data, y_data, point):
    x,y,sy=profiles(x_data, y_data,n_bins=50)
    y_p,uncert_y,_=profiles(x, sy,n_bins=50)
    interp_value=interpolation(x,y,point,kind='linear')
    interp_uncertainty=interpolation(y_p, uncert_y, point)
    
    return interp_value, interp_uncertainty

def predictions_x(x_data, y_data, point):
    x,y,sy=profiles(x_data, y_data)
    interp_value=interpolation_y(x,y,point,kind='linear')
    interp_uncertainty=interpolation_y(y,sy,point,kind='linear')
    
    return interp_value, interp_uncertainty

def MonteCarlo2(radius, distance, particles, desviation):
    '''
    radius: Radius of the detector
    distance: Distance from the place where the desintegration took place to the detector.
    particles: Number of particles simulated.
    desviation: List or array with posible desviations in the (x,y) plane
    ### OUTPUT
    effciency: returns the geometrical efficiency of the detector.
    '''
    def distribucion(n_puntos, r = radius, theta_0 = np.pi):
        
        puntos_unif_theta = np.random.random(n_puntos)
        puntos_unif_phi = np.random.random(n_puntos)
        
        phi_dist = 2*np.pi*puntos_unif_phi
        theta_dist = np.arccos(puntos_unif_theta - 1)
        
        return phi_dist, theta_dist
    
    phi,theta=distribucion(particles)
    particles_detected=0
    particles_lost=0
    
    for i in range(particles):
        x_final=desviation[0]+(distance+desviation[2])*np.tan(theta[i])*np.cos(phi[i])
        y_final=desviation[1]+(distance+desviation[2])*np.tan(theta[i])*np.sin(phi[i])

        if (x_final)**2+(y_final)**2<radius**2:
            particles_detected+=1
        else:
            particles_lost+=1
    
    efficiency=particles_detected/particles
    snd = np.sqrt(particles_detected) 
    snf = np.sqrt(particles_lost)
    seficg = np.sqrt((snd / particles - snd * particles_detected / particles**2)**2 + (snf * particles_detected / particles**2)**2)
    
    return efficiency, seficg

def radius_filter(magnitude_to_filter, R, DT, DT_min, DT_max):
    energías_eventos_corte=[]
    DT_corte=[]
    R_corte=[]
    
    for i in range(len(magnitude_to_filter)):
        if DT_max>DT[i] and DT_min<DT[i] and DT[i]>0:
            energías_eventos_corte.append(magnitude_to_filter[i])
            R_corte.append(R[i])
            DT_corte.append(DT[i])
            
    print(f'El porcentaje de eventos dentro de este corte es {np.round(len(energías_eventos_corte)/len(magnitude_to_filter)*100,3)}%')
    
    return energías_eventos_corte, DT_corte, R_corte

def MonteCarlo(radius, distance, particles, desviation):
    '''
    radius: Radius of the detector
    distance: Distance from the place where the desintegration took place to the detector.
    particles: Number of particles simulated.
    desviation: List or array with posible desviations in the (x,y) plane
    ### OUTPUT
    effciency: returns the geometrical efficiency of the detector.
    '''
    particles_detected=0
    particles_lost=0
    
    for element in range(particles):
        theta = np.arccos(np.random.uniform() - 1)
        phi=np.random.uniform(0,2*np.pi)
        
        x_final=desviation[0]+(distance+desviation[2])*np.tan(theta)*np.cos(phi)
        y_final=desviation[1]+(distance+desviation[2])*np.tan(theta)*np.sin(phi)
        
        if (x_final)**2+(y_final)**2<radius**2:
            particles_detected+=1
        else:
            particles_lost+=1
        
    efficiency=particles_detected/particles
    snd = np.sqrt(particles_detected) 
    snf = np.sqrt(particles_lost)
    seficg = np.sqrt((snd / particles - snd * particles_detected / particles**2)**2 + (snf * particles_detected / particles**2)**2)
    
    return efficiency, seficg

def pull(data, uncertainty, expected_value, plot=False):
    normalized_data=(data-expected_value)/uncertainty
    if plot==True:
        counts, bin_edges = np.histogram(normalized_data.T, bins=25, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  
        errors = np.sqrt(counts) / np.sum(counts)  

        plt.hist(normalized_data.T, bins=25, edgecolor='black', density=True, label='pull')
        plt.errorbar(bin_centers, counts, yerr=errors,fmt='.', color='black')
        
        return bin_centers, counts
    

def histogram_parameters(parameters, bins, plot=False): 
    counts, bin_edges, _ = binned_statistic(parameters, parameters, statistic='count', bins=30)
    errors = np.sqrt(counts)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    def gaussian(x,N,mu,sigma):
        return N*np.exp(-1/2*(x-mu)**2/sigma**2)
    
    popt,pcov=curve_fit(gaussian, bin_centers, counts, p0=[max(counts),bin_centers[np.argmax(counts)],1000])
    if plot==True:
        plt.bar(bin_centers, counts, width=np.diff(bin_edges), edgecolor='black',label='Experimental parameter')
        plt.errorbar(bin_centers, counts, yerr=errors, fmt='none', color='black')
        xplot=np.linspace(min(parameters),max(parameters),1000)
        plt.plot(xplot,gaussian(xplot,*popt),color='red',label='Gaussian')
        plt.legend(loc='best')
        plt.grid(True)
    
    plt.show()
    
    expected_parameter=popt[1]

    print('The best value for the parameter is', expected_parameter,'+-', np.sqrt(pcov[1][1]))
    
    return expected_parameter, np.sqrt(pcov[1][1])

def DT_predictor(x_data,y_data, threshold ,n_bins=50):
    '''
    x_data: array with data in the x-axis (S2w for example)
    y_data: array with data in the y-axis (DT)
    '''
    x,y,sy=profiles(x_data, y_data, threshold, n_bins=n_bins)
    y_p,uncert_y,_=profiles(x, sy, threshold, n_bins=n_bins)
    
    def prediction(x0):
        '''
        x0: value of S2w for which we want to compute the DT.
        ###
        Output:
        array which has as a first component the value for DT predicted and the uncertainty as the second component
        '''
        interp_value=interpolation(x, y, x0, kind='cubic')
        interp_uncertainty=interpolation(y_p, uncert_y, x0)    
    
        return interp_value, interp_uncertainty
    
    return prediction

def histogram_2d(x_data, y_data, z_data, n_bins=50, stat='mean', cmap='jet',xlabel=None, ylabel=None,cbar=None,Mean=None):
    
    mean, x_edges,y_edges,binned = binned_statistic_2d(x_data, y_data, z_data, statistic=stat, bins=n_bins)
    if Mean is not None:
        mean=np.copy(Mean)
        x_edges=np.linspace(min(x_data),max(x_data),n_bins+1)
        y_edges=np.linspace(min(y_data),max(y_data),n_bins+1)
        print(True)
    X, Y = np.meshgrid(x_edges, y_edges)
        
    fig, ax = plt.subplots()  
    c = ax.pcolormesh(X, Y, mean, cmap=cmap,vmin=np.nanmin(mean), vmax=np.nanmax(mean))
    plt.colorbar(c, label=f'{cbar}')
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{ylabel}')
    ax.set_aspect('equal')  
    
    plt.show()
    
    return mean

class Data1d:
    '''
    class which data, you must give (x,y) values so it stores it as data
    '''
    def __init__(self, xdata, ydata, Yrms):
        self.xdata=xdata
        self.ydata=ydata
        self.rms_y=Yrms
        return
    def __str__(self):
        return f'({self.xdata}, {self.ydata})'
    
    def fit(self, function ,p0):
        popt, pcov = curve_fit(function, self.xdata, self.ydata, p0=p0)
        return popt, pcov
    def plot(self):
        return plt.errorbar(self.xdata, self.ydata, self.rms_y, fmt='.', elinewidth=.5, ecolor='black', color = 'red', label='Data points')