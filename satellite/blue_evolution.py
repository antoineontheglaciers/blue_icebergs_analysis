#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:11:40 2024

@author: antoine
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import minimize
from sklearn.metrics import r2_score

plt.rcParams['font.family'] = 'Bitstream Vera Sans'
font = 18
font2 = font-7

def exponential_func(x, a, b, c):
    return a * b*np.exp(-x) + c

path = "/home/antoine/Bureau/post_thesis/post_thesis_data_results/results/"
plt.close('all')
# could be more efficient but good as it is:
    
mean1 = np.load(path + "evol1meanval.npy")
tv1 = [0,2,3,5]
#tv1 = [item + 1 for item in tv1]
blue1 = mean1[1,:] / (mean1[1,:] + mean1[2,:] + mean1[3,:])
T1 = datetime(2000,8,9,14,51)    # First apparition 2017

mean2 = np.load(path + "evol2meanval.npy")
tv2 = [0,2,3,12]
blue2 = mean2[1,:] / (mean2[1,:] + mean2[2,:] + mean2[3,:])
T2 = datetime(2000,5,1,13,25)    # First apparition 2021

mean4 = np.load(path + "evol4meanval.npy")
tv4 = [0,2,5]
blue4 = mean4[1,:] / (mean4[1,:] + mean4[2,:] + mean4[3,:])
T4 = datetime(2000,5,21,15,23)    # First apparition 2020

mean9 = np.load(path + "evol9meanval.npy")
tv9 = [0,3,10,12]
blue9 = mean9[1,:] / (mean9[1,:] + mean9[2,:] + mean9[3,:])
T9 = datetime(2000,7,30,13,15)    # First apparition 2020

mean10 = np.load(path + "evol10meanval.npy")
tv10 = [0,2,3,12]
blue10 = mean10[1,:] / (mean10[1,:] + mean10[2,:] + mean10[3,:])
T10 = datetime(2000,5,1,13,25)    # First apparition 2021

mean16 = np.load(path + "evol16meanval.npy")
tv16 = [0,2,4,9]
blue16 = mean16[1,:] / (mean16[1,:] + mean16[2,:] + mean16[3,:])
T16 = datetime(2000,5,14,13,49)    # First apparition 2021

mean21 = np.load(path + "evol21meanval.npy")
tv21 = [1,3,5,8,9,11,13]  #2&3 are cloudy
blue21 = mean21[1,:] / (mean21[1,:] + mean21[2,:] + mean21[3,:])
T21 = datetime(2000,9,17,10,8)    # First apparition 2021

mean20 = np.load(path + "evol20meanval.npy")
tv20 = [1,3,5,8,9,11,13]  #2&3 are cloudy
blue20 = mean20[1,:] / (mean20[1,:] + mean20[2,:] + mean20[3,:])
T20 = datetime(2000,9,17,10,8)    # First apparition 2021

mean19 = np.load(path + "evol19meanval.npy")
tv19 = [1,3,5,8,9,11,13]  #2&3 are cloudy
blue19 = mean19[1,:] / (mean19[1,:] + mean19[2,:] + mean19[3,:])
T19 = datetime(2000,9,17,10,8)    # First apparition 2021

#==============================================================================
time_vectors = [tv1, tv2, tv4, tv9, tv10, tv16, tv19, tv20, tv21]
y_vectors_sun = [mean1, mean2, mean4, mean9, mean10, mean16]
y_vectors_cloudy = [mean19, mean20, mean21]

#==============================================================================

blue_evol_sun = [blue1, blue2, blue4, blue9, blue10, blue16]
tv_sun = [tv1, tv2, tv4, tv9, tv10, tv16]
T_sun = [T1, T2, T4, T9, T10, T16]

blue_evol_cloudy = [blue19, blue20, blue21]
tv_cloudy = [tv19, tv20, tv21]
T_cloudy = [T19, T20, T21]

evol_sun = np.zeros((6,13))
evol_cloudy = np.zeros((3,14))


#------------------------------------------------------------------------------
for i in range(len(blue_evol_sun)):
    for index, (pos, val) in enumerate(zip(tv_sun[i], blue_evol_sun[i])):
       evol_sun[i,pos] = val

evol_sun[evol_sun == 0] = float('nan')

for i in range(len(blue_evol_cloudy)):
    for index, (pos, val) in enumerate(zip(tv_cloudy[i], blue_evol_cloudy[i])):
       evol_cloudy[i,pos] = val


evol_cloudy[evol_cloudy == 0] = float('nan')

#==============================================================================
# evolution of the blue component

fig, ax = plt.subplots(figsize=(8,5))

# Fit exponential =============================================================

def func(t, a, b, c):
    return a*np.exp(-b*t) + c

# Fit "sun"

evol_sun_nan = evol_sun[~np.isnan(evol_sun)]
x_sun = np.hstack(tv_sun)
x_sun = x_sun + 0.5

y_sun = np.hstack(evol_sun_nan)

p_sun, pcov = curve_fit(func, x_sun, y_sun)

T1 = np.linspace(0,14,100)
T1b = np.linspace(0.5,9,100)

x_sun6 = x_sun[x_sun<6]
y_sun6 = y_sun[x_sun<6]
p_sun6, pcov6s = curve_fit(func, x_sun6, y_sun6)
ax.plot(T1b, func(T1b,p_sun6[0], p_sun6[1], p_sun6[2]), linewidth = 3, color = 'black', label = '$ 0.31 \cdot e^{- 1.75 \cdot t} + 0.35 $', zorder = 0)
y_pred = func(x_sun6, *p_sun6)
r2_sun = r2_score(y_sun6, y_pred)

# Fit "cloudy"

evol_cl_nan = evol_cloudy[~np.isnan(evol_cloudy)]
x_cl = np.hstack(tv_cloudy)
x_cl = x_cl+0.5

y_cl = np.hstack(evol_cl_nan)

p_cl, pcov = curve_fit(func, x_cl, y_cl)

T2 = np.linspace(1,14,100)
T2b = np.linspace(1.5,9,100)
#ax.plot(T2 + 0.5, func(T2,p_cl[0], p_cl[1], p_cl[2]), linewidth = 3, color = 'orange', label = '$ p_0 \cdot e^{p_1 \cdot (t-0.5)} + p_3 $', zorder = 0)

x_cl6 = x_cl[x_cl<9]
#x_cl6 = np.array([item -1 for item in x_cl6])
y_cl6 = y_cl[x_cl<9]
p_cl6, pcov6c = curve_fit(func, x_cl6, y_cl6)
ax.plot(T2b, func(T2b,p_cl6[0], p_cl6[1], p_cl6[2]), linewidth = 3, color = 'black',linestyle ='--', label = '$ 0.37 \cdot e^{- 0.68 \cdot t} + 0.35 $', zorder = 0)
y_predc = func(x_cl6, *p_cl6)
r2_cl = r2_score(y_cl6, y_predc)

# Fit all

x_all = np.concatenate((x_sun,x_cl))
y_all = np.concatenate((y_sun,y_cl))

p_all, pcov = curve_fit(func, x_all, y_all)

T3 = np.linspace(0,14,100)
#ax.plot(T3 + 0.5, func(T3,p_all[0], p_all[1], p_all[2]), linewidth = 3, color = 'green', zorder = 0)


# Plot the points

for row in range(np.shape(evol_sun)[0]):
    if row == 0:
        ax.scatter(np.linspace(0.5,12.5,13), evol_sun[row,:], color = 'black', label = "high solar radiation", zorder = 1)
        ax.errorbar([0.5], evol_sun[row,0], color = 'black', xerr = 0.5)
    else:
        ax.scatter(np.linspace(0.5,12.5,13), evol_sun[row,:], color = 'black', zorder = 1)
        ax.errorbar([0.5], evol_sun[row,0], color = 'black', xerr = 0.5)

#ax.plot(T1 + 0.5, func(T1,p_sun[0], p_sun[1], p_sun[2]), linewidth = 3, color = 'yellow', label = '$ p_0 \cdot e^{p_1 \cdot (t-0.5)} + p_3 $', zorder = 0)
        
for row in range(np.shape(evol_cloudy)[0]):
    if row == 0:
        ax.scatter(np.linspace(0.5,13.5,14), evol_cloudy[row,:], facecolors='none', edgecolors='black', label = "low solar radiation", zorder = 1)
        ax.errorbar([1.5], evol_cloudy[row,1], color = 'black', xerr = 0.5)
    else:
        ax.scatter(np.linspace(0.5,13.5,14), evol_cloudy[row,:], facecolors='none', edgecolors='black', zorder = 1)
        ax.errorbar([1.5], evol_cloudy[row,1], color = 'black', xerr = 0.5)

#ax.plot(T2 + 0.5, func(T2,p_cl[0], p_cl[1], p_cl[2]), linewidth = 3, color = 'orange', label = '$ p_0 \cdot e^{p_1 \cdot (t-0.5)} + p_3 $', zorder = 0)
    
# plot the mean of rgb ratios -------------------------------------------------
x_vec = [0.2,0.8]
x_vec = [-1, 14]
ax.plot(x_vec, [0.457]*2, linewidth =3, color = 'blue', label= "Blue icebergs mean", zorder = -1) 
# plot ratio from antarctica
ax.plot(x_vec,[0.373]*2, linewidth = 3, color = 'turquoise', label = "Blue ice Antarctica", zorder = -1)

ax.plot([-1,14], [0.358,0.358], linewidth = 3, color ='grey', label = "White icebergs mean", zorder = -1)

# plot glacier ice 
#ax.plot(x_vec,[0.34]*2, linewidth = 2, color = 'violet', label = "Glacier ice")
ax.plot(x_vec,[0.34]*2, linewidth = 3, color = 'red', label = "Clean bare ice", zorder = -1)


#--------------------------------------------------------------------------------

print("Half time values:\n")
t_sun = -np.log(0.5)/p_sun6[1]
t_cl = -np.log(0.5)/p_cl6[1]

print("t12_sun =", t_sun)
print("t12_cl =", t_cl,"\n")




# PLOT THE EXPONENTIAL CURVES FITTED:

res_sunny = np.load(path + "res_sunny.npy")
res_cloudy = np.load(path + "res_cloudy.npy" )

x1 = np.linspace(0,12,121)
#ax.plot(x1+0.5, exponential_func(x1, res_sunny[0], res_sunny[1], res_sunny[2]), color = 'black', label = r"$ f(t) = 0.32 \cdot 0.41^{t-0.5} + 0.34$", zorder = 0)
""" 
x2 = np.linspace(0,13,131)
ax.plot(x2, exponential_func(x2, res_cloudy[0], res_cloudy[1], res_cloudy[2]), color = 'black', alpha = 0.5)
"""

x3 = np.linspace(0,12,121)
res_cloudy_2 = np.load(path + "res_cloudy_2.npy" )
#ax.plot(x3+1.5,exponential_func(x3, res_cloudy_2[0], res_cloudy_2[1], res_cloudy_2[2]), color = 'black', linestyle = '--', label = r"$ f(t) = 0.28 \cdot 0.48^{t-1.5} + 0.36$", zorder = 0) 
        
ax.set_xlabel("Days after calving",fontsize = font)
ax.set_ylabel(r"Blue color index: $\frac{b}{r+g+b}$",fontsize = font)


ax.set_xticks(np.linspace(0,14,8), [str(int(item)) for item in np.linspace(0,14,8)])
ax.tick_params(axis='both', which='major', labelsize=font)
#ax.plot([-1,14], [0.33,0.33], color ='black', alpha = 0.5, linestyle = "dashdot")
ax.set_xlim(0,9)
#ax.set_xlim(0,14)
y_ticks = [0.35, 0.40, 0.45, 0.50]
ax.set_yticks(y_ticks, ["0.35", "0.40", "0.45", "0.50"])
handles, labels = ax.get_legend_handles_labels()
order = [2,0,3,1,4,5,6,7]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],handlelength = 5)

#==============================================================================
"""
fig2, ax1 = plt.subplots()

# Plot the points


for row in range(np.shape(evol_sun)[0]):
    if row == 0:
        ax1.scatter(np.linspace(0.5,12.5,13), evol_sun[row,:], color = 'black', label = "high solar radiation")
        ax1.plot([item + 0.5 for item in tv_sun[row]], blue_evol_sun[row], color = 'black')
        #ax1.errorbar([0.5], evol_sun[row,0], color = 'black', xerr = 0.5)
    else:
        ax1.scatter(np.linspace(0.5,12.5,13), evol_sun[row,:], color = 'black')
        ax1.plot([item + 0.5 for item in tv_sun[row]], blue_evol_sun[row], color = 'black')
        #ax1.errorbar([0.5], evol_sun[row,0], color = 'black', xerr = 0.5)
        
for row in range(np.shape(evol_cloudy)[0]):
    if row == 0:
        ax1.scatter(np.linspace(0.5,13.5,14), evol_cloudy[row,:], facecolors='none', edgecolors='black', label = "low solar radiation")
        ax1.plot([item + 0.5 for item in tv_cloudy[row]], blue_evol_cloudy[row], color = 'red')
        #ax1.errorbar([1.5], evol_cloudy[row,1], color = 'black', xerr = 0.5)
    else:
        ax1.scatter(np.linspace(0.5,13.5,14), evol_cloudy[row,:], facecolors='none', edgecolors='black')
        ax1.plot([item + 0.5 for item in tv_cloudy[row]], blue_evol_cloudy[row], color = 'red')
        #ax1.errorbar([1.5], evol_cloudy[row,1], color = 'black', xerr = 0.5)
        
#ax1.set_xscale('log')
ax1.set_xlabel("Discretized days after calving",fontsize = font)
ax1.set_ylabel(r"Blue color index: $\frac{b}{r+g+b}$",fontsize = font)

# Calculate the decrease
# But only if 2 days apart (to make them comparable)

Rs = np.zeros(len(tv_sun))

for i,tv in enumerate(tv_sun):
    
    sv = blue_evol_sun[i] #single vector
    Rs[i] = (sv[1] - sv[0]) / (tv[1] - tv[0])
    
Rc = np.zeros(len(tv_cloudy))

for i,tv in enumerate(tv_cloudy):
    
    sv = blue_evol_cloudy[i] #single vector
    Rc[i] = (sv[1] - sv[0]) / (tv[1] - tv[0])
"""
    
"""
fig3, ax3 = plt.subplots()

ax3.scatter(T_sun, Rs, color = 'red')
ax3.scatter(T_cloudy, Rc, color = 'blue')
"""

"""
fig, ax = plt.subplots()

blue1 = mean1[1,:] / (mean1[1,:] + mean1[2,:] + mean1[3,:])
ax.plot(tv1, blue1)

blue2 = mean2[1,:] / (mean2[1,:] + mean2[2,:] + mean2[3,:])
ax.plot(tv2, blue2)

blue4 = mean4[1,:] / (mean4[1,:] + mean4[2,:] + mean4[3,:])
ax.plot(tv4, blue4)

blue9 = mean9[1,:] / (mean9[1,:] + mean9[2,:] + mean9[3,:])
ax.plot(tv9, blue9)

blue10 = mean10[1,:] / (mean10[1,:] + mean10[2,:] + mean10[3,:])
ax.plot(tv10, blue10)

blue16 = mean16[1,:] / (mean16[1,:] + mean16[2,:] + mean16[3,:])
ax.plot(tv16, blue16)

blue21 = mean21[1,:] / (mean21[1,:] + mean21[2,:] + mean21[3,:])
ax.plot(tv21, blue21, color='blue', linewidth = 3)

blue20 = mean20[1,:] / (mean20[1,:] + mean20[2,:] + mean20[3,:])
ax.plot(tv20, blue20, color='red', linewidth = 3)

blue19 = mean19[1,:] / (mean19[1,:] + mean19[2,:] + mean19[3,:])
ax.plot(tv19, blue19, color='red', linewidth = 3)

ax.set_ylabel("b/(r + g + b)")
ax.set_xlabel("Days after calving")

fig2, ax2 = plt.subplots()

ax2.scatter(tv1, blue1)

ax2.scatter(tv2, blue2)

ax2.scatter(tv4, blue4)

ax2.scatter(tv9, blue9)

ax2.scatter(tv10, blue10)

ax2.scatter(tv16, blue16)

ax2.scatter(tv21, blue21, color='blue', linewidth = 4)

ax2.scatter(tv20, blue20, color='red', linewidth = 4)

ax2.scatter(tv19, blue19, color='green', linewidth = 4)

ax2.set_ylabel("b/(r + g + b)")
ax2.set_xlabel("Days after calving")

"""

#==============================================================================

# Fit an exponential curve

"""
ax0 = axs[0]
ax0.plot(tv1, mean1.T, linewidth = 3)
ax0.plot(tv1, mean1[0,:], color ='white'
"""


#==============================================================================
# Fit exponential:
"""
def compute_rmse(x,res,evol_mat):
    tot_sum = 0
    for ssx in x:
        sx = int(ssx)
        print(sx)
        sum_x = np.sqrt(np.nansum((evol_mat[:,sx]-res[sx])**2) / np.count_nonzero(~np.isnan(evol_mat[:,sx]-res[sx])))
        tot_sum += sum_x
        
    return tot_sum
"""

def compute_rmse(x, res, evol_mat):
    """
    Compute the total RMSE for selected columns in evol_mat compared to res.

    Parameters:
    x (list of int): List of indices of columns to be considered.
    res (np.array): Reference array with the same length as the number of columns in evol_mat.
    evol_mat (np.array): 2D array where RMSE is computed for selected columns.

    Returns:
    float: The total RMSE for the selected columns.
    """
    tot_sum = 0
    for sx in x:
        # Convert to integer, if necessary
        sx = int(sx)
        
        # Compute the squared differences while handling NaNs
        differences = evol_mat[:, sx] - res[sx]
        squared_differences = np.square(differences)
        squared_differences = np.nan_to_num(squared_differences, nan=0.0)
        
        # Compute the mean squared error, ignoring NaNs
        count_non_nan = np.count_nonzero(~np.isnan(differences))
        if count_non_nan > 0:
            mean_squared_error = np.sum(squared_differences) / count_non_nan
            rmse = np.sqrt(mean_squared_error)
            tot_sum += rmse
            
    return tot_sum

"""    
def compute_rmse_2(x,res,evol_mat):
    tot_sum = 0
    for ssx in x:
        sx = int(ssx)
        print(sx)
        sum_x = np.nansum(abs(evol_mat[:,sx]-res[sx]))
        tot_sum += sum_x
        
    return tot_sum
"""
# SUNNY
"""
a = np.linspace(0.275,0.325,51)
b = np.linspace(0.375,0.425,51)
c = np.linspace(0.32,0.38,51)
prev_tot_sum = 5

for sa in a:
    for sb in b:
        for sc in c:
            x = np.linspace(0,12,13)
            res = exponential_func(x, sa, sb, sc)
            
            tot_sum = compute_rmse(x,res, evol_sun)
            print("tot_sum:", tot_sum)
            print("prev_tot_sum:", prev_tot_sum)
            
            if tot_sum < prev_tot_sum:
                print("New best values !")
                best_a_s = sa
                best_b_s = sb
                best_c_s = sc
                prev_tot_sum = tot_sum
        
ax.plot(x, exponential_func(x, best_a_s, best_b_s, best_c_s), color = 'black', alpha = 0.8)     
res_sun = [best_a_s, best_b_s, best_c_s] 
np.save(path + "res_sun.npy", res_sun)
"""
"""
# CLOUDY
a = np.linspace(0.3,0.6,51)
b = np.linspace(0.5,1,51)
c = np.linspace(0.3,0.5,51)
prev_tot_sum = 5

for sa in a:
    for sb in b:
        for sc in c:
            x = np.linspace(0,13,14)
            res = exponential_func(x, sa, sb, sc)
            
            tot_sum = compute_rmse(x,res, evol_cloudy)
            print("tot_sum:", tot_sum)
            print("prev_tot_sum:", prev_tot_sum)
            
            if tot_sum < prev_tot_sum:
                print("New best values !")
                best_a_c = sa
                best_b_c = sb
                best_c_c = sc
                prev_tot_sum = tot_sum
        
ax.plot(x, exponential_func(x[1,], best_a_c, best_b_c, best_c_c), color = 'black',linestyle = '--', alpha = 0.8)        
"""
"""
# CLOUDY 2
a = np.linspace(0.1,0.3,51)
b = np.linspace(0.3,0.5,51)
c = np.linspace(0.3,0.4,21)
prev_tot_sum = 5

evol_cloudy_2 = evol_cloudy[:,1:]

for sa in a:
    for sb in b:
        for sc in c:
            x = np.linspace(0,12,13)
            res = exponential_func(x, sa, sb, sc)
            
            tot_sum = compute_rmse(x,res, evol_cloudy_2)
            print("tot_sum:", tot_sum)
            print("prev_tot_sum:", prev_tot_sum)
            
            if tot_sum < prev_tot_sum:
                print("New best values !")
                best_a_c2 = sa
                best_b_c2 = sb
                best_c_c2 = sc
                prev_tot_sum = tot_sum

res_cloudy_2 = [best_a_c2, best_b_c2, best_c_c2]
np.save(path + "res_cloudy_2", res_cloudy_2)
"""