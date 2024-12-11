# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:58:08 2024

@author: pablo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

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
        if DT_max>DT[i] and DT_min<DT[i]:
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
        
# def Neuronal_Network(x_data, y_data, value):
    
#     # Prepare data
#     x = y_data.to_numpy().reshape(-1, 1)
#     y = x_data.to_numpy().reshape(-1, 1)
#     X_scaler = MinMaxScaler()
#     y_scaler = MinMaxScaler()
#     X = X_scaler.fit_transform(x)
#     Y = y_scaler.fit_transform(y)

#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(units=1, activation='linear', input_shape=[1]),
#         tf.keras.layers.Dense(units=64, activation='softplus'),  # Reduced units
#         tf.keras.layers.Dense(units=128, activation='softplus'),
#         tf.keras.layers.Dense(units=64, activation='softplus'),# Reduced units
#         tf.keras.layers.Dense(units=1, activation='linear')
#     ])
    
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')
    
#     # Training with a larger batch size and early stopping
#     print("Comenzando entrenamiento...")
#     historial = model.fit(X, Y, epochs=1000, batch_size=1024, verbose=False)
#     print("Modelo entrenado")
    
#     # Plot loss curve
#     plt.xlabel('#Epoch')
#     plt.ylabel('Loss')
#     plt.plot(historial.history["loss"])
#     value_array = np.array([[value]])
#     # Make predictions
#     print('Hagamos unas predicciones')
#     resultado = X_scaler.inverse_transform(model.predict(value_array))
#     print(resultado)
#     return resultado
