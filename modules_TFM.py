# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:58:08 2024

@author: pablo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d

def profiles(xdata,ydata,n_bins=50,x_label=None, y_label=None, plot=False):
    '''
    xdata: np.array or 1 column dataframe with data for the x-axes
    ydata: np.array or 1 column dataframe with data for the x-axes
    n_bins: must be an integer, number of bins for the histogram. Default value: 50 could change if you had poor statistic
    x_label: string that contains the name of the magnitude in the x axes
    y_label: string that contains the name of the magnitude in the y axes
    ######
    Output: returns a plot with the profile of ydata as a function of xdata. This profile is superposed over an histogram of the number of events so we can check if the statistic is good enough to consider one point as relevant.
    '''        
    ydata_filtered = np.array([s for s, d in zip(ydata, xdata) if d > 10])
    xdata_filtered = np.array([i for i in xdata if i> 10])
    
    mean_y, bin_edges, _ = binned_statistic(xdata_filtered, ydata_filtered, statistic='mean', bins=n_bins)
    std_y, _, _ = binned_statistic(xdata_filtered, ydata_filtered, statistic='std', bins=n_bins)
    count, _, _ = binned_statistic(xdata_filtered, ydata_filtered, statistic='count', bins=n_bins)

    if mean_y.any()<0: print(True)
    else: print(False)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    uncertainty = std_y / np.sqrt(count)
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
        plt.plot(mean_y, uncertainty, 'b.')
        plt.ylabel(f'Uncertainty in {y_label}')
        plt.xlabel(f'{y_label}')
        plt.grid(True)
        
        plt.show()
    
    return bin_centers, mean_y
    
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
        if Radial_position[i]<=cut:
            central_events_energy.append(Energy[i])
            axial_position_central_events.append(DT[i])
        else:
            external_events_energy.append(Energy[i])
            axial_position_external_events.append(DT[i])
            
    print(f'The percentage of events insode the cut of {cut}mm is {np.round(len(axial_position_central_events)/len(DT)*100,3)}%')
    print(f'Then , the percentage outside the cut must be {np.round(len(axial_position_external_events)/len(DT)*100,3)}%')
    
    return central_events_energy,  axial_position_central_events, external_events_energy, axial_position_external_events

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
    interpolation_function=interp1d(y_data, x_data, kind, fill_value="extrapolate")
    value=interpolation_function(point)
    print(f"Interpolated value at y={point} using interpolation: {value}")
    
    return value

def MonteCarlo(radius, distance, particles, desviation):
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
        
        x = np.sin(theta_dist)*np.cos(phi_dist)
        y = np.sin(theta_dist)*np.sin(phi_dist)
        z = np.cos(theta_dist)
        
        return x, y, z, phi_dist, theta_dist
    
    x,y,z,phi,theta=distribucion(particles)
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
    return efficiency

