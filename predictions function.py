# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:13:25 2025

@author: pablo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d

print('Just with the last functor is enough to predict, the other 2 functions are there cause they are used in DT_predictor')

def profiles(xdata,ydata, threshold, n_bins=50,x_label=None, y_label=None, plot=False):
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
    
    return bin_centers, mean_y, uncertainty

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

def DT_predictor(x_data,y_data, threshold ,n_bins=50):
    '''
    x_data: array with data in the x-axis (S2w for example)
    y_data: array with data in the y-axis (DT)
    '''
    x,y,sy=profiles(x_data, y_data, threshold, n_bins=50)
    y_p,uncert_y,_=profiles(x, sy, threshold, n_bins=50)
    
    def prediction(x0):
        '''
        x0: value of S2w for which we want to compute the DT.
        ###
        Output:
        array which has as a first component the value for DT predicted and the uncertainty as the second component
        '''
        interp_value=interpolation(x, y, x0, kind='linear')
        interp_uncertainty=interpolation(y_p, uncert_y, x0)    
    
        return interp_value, interp_uncertainty
    
    return prediction