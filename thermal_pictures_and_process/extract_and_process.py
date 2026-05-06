#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:44:45 2023

@author: antoine
"""

# extract a thermal image with exiftool

import numpy as np
import pylab as plt
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image
import os
import subprocess
import sys
import tifffile as tiff
import imageio 
import matplotlib


#==============================================================================

# CHOOSE IMAGE AND PARAMETERS (OR TAKE THOSE FROM THE IMAGES UNCOMMENTING BELOW)

target_DC = 2736

gen_path = '.../thermal_pictures_and_process'


appReflTemp_C = -30   
objEmissivity = 0.98         
airTemp_C =  8           
airRelHumidity_perc = 65     
objDistance_m = 3500        

# CALIBRATION OF EMISSIVITY ON KNOWN T°

calibration = True
opt_T = 0                # used to calibrate emissivity values
#==============================================================================

#y_test,x_test = 134,125

filename = gen_path + "IR_" + str(target_DC-1) + '.jpg'
DC = gen_path + "DC_" + str(target_DC) + ".jpg"
dc_img = Image.open(DC)

#==============================================================================

# POLICE STYLE

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
#matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

#==============================================================================

#==============================================================================

def bytetoint(b):
    val = (b-b%256)/256 + b%256*256
    return val
# import subprocess
# res = subprocess.run(['exiftool', filename], capture_output=True, text=True)


#==============================================================================

# EXTRACT THE METADATA

if os.path.exists(filename[:-3]+'tif'):
    os.remove(filename[:-3]+'tif')
    
cmd = f'exiftool {filename}'
res = os.popen(cmd).read()
info = dict()
for line in res.splitlines():
    l, r = line.split(':', 1)
    info[l.strip()] = r


"""
float(info["Reflected Apparent Temperature"][:-2])
float(info["Emissivity"])
float(info["Atmospheric Temperature"][:-2])
float(info["Relative Humidity"][:-2])
float(info["Object Distance"][:-2])
"""

#==============================================================================

# COMPUTE UNCERTAINTY ON TEMPATURE
"""
appReflTemp_C = ufloat(-30,10)           #float(info["Reflected Apparent Temperature"][:-2])
objEmissivity = 0.98            #float(info["Emissivity"])
airTemp_C = ufloat(10,5)               #float(info["Atmospheric Temperature"][:-2])
airRelHumidity_perc = ufloat(65,20)      #float(info["Relative Humidity"][:-2])
objDistance_m = ufloat(3000, 1000)          #float(info["Object Distance"][:-2])
"""
params = [appReflTemp_C, objEmissivity, airTemp_C, airRelHumidity_perc,objDistance_m]
np.save("thermal_pictures_and_process/RAW_Visualisation_" + str(target_DC) + "_params.npy", params)

atmAlpha1 = float(info["Atmospheric Trans Alpha 1"])
atmAlpha2 = float(info["Atmospheric Trans Alpha 2"])
atmBeta1 = float(info["Atmospheric Trans Beta 1"])
atmBeta2 = float(info["Atmospheric Trans Beta 2"])
atmX = float(info["Atmospheric Trans X"])

sensorG = float(info["Planck R1"])
sensorB = float(info["Planck B"])
sensorF = float(info["Planck F"])
sensorO = float(info["Planck O"])
sensorR = float(info["Planck R2"])


# EXTRACT THE THERMAL IMAGE
cmd = 'exiftool {} -RawThermalImage -b -w {}'.format(filename, 'tif')
os.system(cmd)

# EXTRACT PARAMETERS
"""
dirExifTool = ".../Image-ExifTool-12.67"

parameters = [
    dirExifTool + "/exiftool",
    "-Planck*",
    "-Atmospheric*",
    "-Reflected*",
    "-Emissivity*",
    "-Relative*",
    "-Object*",
    "-RawThermalImageType",
    "-CameraModel",
    "-DateTimeOriginal",
    filename
]

result = subprocess.run(parameters, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True)

if result.returncode == 0:
    print("ExifTool command was successful.")
    print("Output:\n", result.stdout)
else:
    print("ExifTool command failed with error:")
    print("Error:\n", result.stderr)
"""




#====================== FLIR ALGORITHM ========================================

# READ THE INFRARED FILE
img = mpimg.imread(filename[:-3]+'tif')

# CALCULATE DERIVED VARIABLES

appReflTemp_K = appReflTemp_C + 273.15
airTemp_K = airTemp_C + 273.15
airWaterContent = airRelHumidity_perc /100 * np.exp(1.5587 + 0.06939 * airTemp_C - 0.00027816 * airTemp_C**2 + 0.000000068455 * airTemp_C**3)
atmTau = atmX * np.exp(-np.sqrt(objDistance_m) * (atmAlpha1 + atmBeta1 * np.sqrt(airWaterContent))) + (1-atmX) * np.exp(-np.sqrt(objDistance_m) * (atmAlpha2 + atmBeta2 * np.sqrt(airWaterContent)))
atmRawSignal_DN = sensorG/(sensorR*(np.exp(sensorB/(airTemp_K))-sensorF)) - sensorO
reflRawSignal_DN = sensorG/(sensorR*(np.exp(sensorB/(appReflTemp_K))-sensorF)) - sensorO


# CALCULATE TEMPERATURE VALUES FOR EACH PIXEL

rawSignal_DN = img #(img - (img%256))/256 + (img%256)*256 #--> it is extracted from bytes
# previous code ws something with endianness tht is not used anymore

objRawSignal_DN = (rawSignal_DN-atmRawSignal_DN*(1-atmTau)-reflRawSignal_DN*(1-objEmissivity)*atmTau)/objEmissivity/atmTau
objTemp_C = sensorB/np.log(sensorG/(sensorR*(objRawSignal_DN+sensorO))+sensorF)-273.15

np.save(gen_path + "DC_" + str(target_DC) + "numpy.npy", objTemp_C )
#file_path = 'thermal_pictures_and_process/' + filename[-11:-3] + 'tif'
#imageio.imwrite(file_path, objTemp_C)

plt.figure("Plot temp for jpg")
#t_min = -30
#t_max = 10
#norm = plt.Normalize(vmin=t_min,vmax=t_max)

plt.imshow(objTemp_C, cmap = plt.cm.jet)#, norm = norm)
cbar = plt.colorbar()
plt.savefig('thermal_pictures_and_process/' + filename[-11:-3] + 'tif')

#====================== CALIBRATION ===========================================

# PLOT A MAP OF THE NEEDED EMISSIVITIES TO ACHIEVE T_wanted ICE

if calibration == True:
    
    size = np.shape(img)
    opt_e = np.zeros(np.shape(img))
    emissivities = np.linspace(0.05,1,101)
    
    
    for j in range(0,size[0],1):
        for k in range(0,size[1],1):
            
            pixel = img[j,k]
            min_temp = 2
            
            for objEmissivity in emissivities:
            
                objRawSignal_DN = (pixel-atmRawSignal_DN*(1-atmTau)-reflRawSignal_DN*(1-objEmissivity)*atmTau)/objEmissivity/atmTau
                TEMP = sensorB/np.log(sensorG/(sensorR*(objRawSignal_DN+sensorO))+sensorF)-273.15
                
                if abs(TEMP-opt_T) < min_temp and abs(TEMP-opt_T)<0.2:
                    min_temp = abs(TEMP)
                    opt_e[j,k] = objEmissivity
            
    
    
    plt.figure("Emissivity")
    plt.title(f"T_atm={airTemp_C} degC, RH ={airRelHumidity_perc} %, T_refl ={appReflTemp_C} degC, d={objDistance_m} m")
    plt.imshow(opt_e)
    plt.colorbar(label = "Emisssivity (-)")
    tiff.imwrite(f'/thermal_pictures_and_process/emissivity_calibrated_{opt_T}.tif', opt_e)

#==============================================================================

# SEE THE IMPACT OF DISTANCE

# CALCULATE DERIVED VARIABLES

"""
#objDistance_m = np.linspace(2000,6000,4001)
#airRelHumidity_perc = np.linspace(50,100,51)
airTemp_C = np.linspace(0,20,21)
appReflTemp_K = appReflTemp_C + 273.15
airTemp_K = airTemp_C + 273.15
airWaterContent = airRelHumidity_perc /100 * np.exp(1.5587 + 0.06939 * airTemp_C - 0.00027816 * airTemp_C**2 + 0.000000068455 * airTemp_C**3)
atmTau = atmX * np.exp(-np.sqrt(objDistance_m) * (atmAlpha1 + atmBeta1 * np.sqrt(airWaterContent))) + (1-atmX) * np.exp(-np.sqrt(objDistance_m) * (atmAlpha2 + atmBeta2 * np.sqrt(airWaterContent)))
atmRawSignal_DN = sensorG/(sensorR*(np.exp(sensorB/(airTemp_K))-sensorF)) - sensorO
reflRawSignal_DN = sensorG/(sensorR*(np.exp(sensorB/(appReflTemp_K))-sensorF)) - sensorO
"""

# CALCULATE TEMPERATURE VALUES FOR EACH PIXEL

"""
rawSignal_DN = img #(img - (img%256))/256 + (img%256)*256 #--> it is extracted from bytes
# previous code ws something with endianness tht is not used anymore

rawSignal_vect = [12000,13000,14000,15000,16000]
plt.figure("Distance influence")

for rawSignal in rawSignal_vect:
    objRawSignal_DN_dist = (rawSignal-atmRawSignal_DN*(1-atmTau)-reflRawSignal_DN*(1-objEmissivity)*atmTau)/objEmissivity/atmTau

    objTemp_C_dist = sensorB/np.log(sensorG/(sensorR*(objRawSignal_DN_dist+sensorO))+sensorF)-273.15

    plt.plot(airTemp_C, objTemp_C_dist)
    
plt.xlabel("Air temperature (°C)")
plt.ylabel("Temperature (°C)")
#plt.xticks(objDistance_m,str(objDistance_m))
plt.show()
"""
#==============================================================================
"""
# CALCULATE TEMPERATURE VALUES FOR EACH PIXEL

rawSignal_DN = img #(img - (img%256))/256 + (img%256)*256 #--> it is extracted from bytes
# previous code ws something with endianness tht is not used anymore

objRawSignal_DN = (rawSignal_DN-atmRawSignal_DN*(1-atmTau)-reflRawSignal_DN*(1-objEmissivity)*atmTau)/objEmissivity/atmTau
objTemp_C = sensorB/np.log(sensorG/(sensorR*(objRawSignal_DN+sensorO))+sensorF)-273.15
"""

# CALCULATE ALTERNATIVE TEMPERATURE WITH EQUATION FROM THE PAPER (obsolete)
"""
objRawSignal_DN2 = (rawSignal_DN-atmRawSignal_DN*(1-atmTau)-reflRawSignal_DN*(1-objEmissivity)*atmTau)/atmTau
objTemp_C2 = sensorB/np.log(sensorG*objEmissivity/(sensorR*(objRawSignal_DN+sensorO))+sensorF)-273.15

# PLOT THE TWO WAYS OF DERIVING TEMPERATURE

plt.figure(1)

plt.subplot(2,2,1)
plt.imshow(img, cmap=plt.cm.jet)
plt.title("RAW DATA")
plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(objTemp_C2, cmap=plt.cm.jet)
plt.title("IRimage paper")
plt.colorbar()

plt.subplot(2,2,4)
plt.title("code-paper")
plt.imshow(objTemp_C - objTemp_C2, cmap='Greys')
plt.colorbar()

plt.subplot(2,2,2)
plt.title("IRimage code")
plt.imshow(objTemp_C, cmap=plt.cm.jet)
plt.colorbar()

plt.show()
"""




# SEE THE INFLUENCE OF DISTANCE AND RELATIVE HUMIDITY
"""
objDistance_ = np.linspace(0,5000,101)
airRelHumidity_perc_ = np.linspace(0,100,101)

airWaterContent = airRelHumidity_perc_ /100 * np.exp(1.5587 + 0.06939 * airTemp_C - 0.00027816 * airTemp_C**2 + 0.000000068455 * airTemp_C**3)

temp2 = np.zeros((101,101))
tau_matrix = np.zeros((101,101))

for i, objDistance__ in enumerate(objDistance_):
    
    atmTau = atmX * np.exp(-np.sqrt(objDistance__) * (atmAlpha1 + atmBeta1 * np.sqrt(airWaterContent))) + (1-atmX) * np.exp(-np.sqrt(objDistance__) * (atmAlpha2 + atmBeta2 * np.sqrt(airWaterContent)))
    objRawSignal_DN__ = (rawSignal_DN[132,192]-atmRawSignal_DN*(1-atmTau)-reflRawSignal_DN*(1-objEmissivity)*atmTau)/objEmissivity/atmTau
    temp2[i,:] = sensorB/np.log(sensorG/(sensorR*(objRawSignal_DN__+sensorO))+sensorF)-273.15
    tau_matrix[i,:] = atmTau
    
plt.figure('Air relative humidity and distance')
plt.xlabel("Distance")
plt.ylabel("Air relative humidity")

N = 11
x_mark =  np.zeros(N)
x_labels = []

for i in range(N):
    x_mark[i] = i*10
    x_labels.append(str(i*50))
    
plt.xticks(x_mark, x_labels)

temp_min = -2
temp_max = 2
#norm = plt.Normalize(vmin=temp_min,vmax=temp_max)

x,y = np.meshgrid(airRelHumidity_perc_,objDistance_)
plt.imshow(temp2)#, norm = norm)
plt.colorbar()

"""


# PLOT RAW DATA, IR IMAGE, COMPUTED TEMPERATURE AND DIFFERENCE BETWEEN BOTH

"""
plt.figure(1)

plt.subplot(2,2,1)
plt.imshow(img, cmap=plt.cm.jet)
plt.title("RAW DATA")
plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(data, cmap=plt.cm.jet)
plt.title("IRimage output")
plt.colorbar()

plt.subplot(2,2,4)
plt.title("Difference owncode - IRimage")
plt.imshow(objTemp_C - data, cmap='Greys')
plt.colorbar()

plt.subplot(2,2,2)
plt.title("Temperature")
plt.imshow(objTemp_C, cmap=plt.cm.jet)
plt.colorbar()

plt.show()

"""


# PLOT ONLY RAW DATA, VISIBLE AND TEMPERATURE



plt.figure(f'DC_{target_DC}_Rawdata_temp_and_visible')

plt.subplot(2,2,4)
x = 0.5
plt.text(x,1, f"objDistance = {objDistance_m}")
plt.text(x,2, f"objEmissivity = {objEmissivity}")
plt.text(x,3, f"airRelHumidity = {airRelHumidity_perc}")
plt.text(x,4, f"airTemp_C =  {airTemp_C}")
plt.text(x,5, f"appReflTemp_C = {appReflTemp_C}")
plt.ylim(-0.5,5.5)
plt.xlim(0,2)
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(img, cmap=plt.cm.jet)
plt.title("RAW DATA")
plt.colorbar()

plt.subplot(2,2,2)
plt.title("Temperature [°C]")
plt.imshow(objTemp_C, cmap=plt.cm.jet)#,vmin=-10, vmax = 2)
plt.colorbar()

plt.subplot(2,2,1)
plt.title("Visible")
plt.imshow(dc_img)


plt.show()


# SEE THE INFLUENCE OF EMISSIVITY

"""

x = np.linspace(0.4, 1,101)
M = len(x)
objRawSignal_DNind = np.zeros(M)
temp = np.zeros(M)



first_member = np.zeros(M)
second_member= np.zeros(M)
third_member = np.zeros(M)

for i in range(M):

    e = x[i]
    objRawSignal_DNind[i] = (rawSignal_DN[125,146]-atmRawSignal_DN*(1-atmTau)-reflRawSignal_DN*(1-e)*atmTau)/e/atmTau 
    first_member[i] = atmRawSignal_DN*(1-atmTau)
    second_member[i] = reflRawSignal_DN*(1-e)*atmTau
    #third_member[i] =
    temp[i] = sensorB/np.log(sensorG/(sensorR*(objRawSignal_DNind[i]+sensorO))+sensorF)-273.15;

    
plt.figure(2)
ax1 = plt.gca()
ax1.set_xlabel("Emissivity")
ax1.set_ylabel("Temperature [°C]")
ax1.plot(x,temp, color="blue", label = "Corrected temperature")
plt.legend()

ax2 = ax1.twinx()
ax2.plot(x,objRawSignal_DNind, color='red', label = 'Corrected Raw Signal')
ax2.set_ylabel("DN")

ax2.plot(x, first_member, color = 'green', label = 'atmRawSignal correction')
ax2.plot(x,second_member, color = 'yellow', label = 'reflRawSignal correction')
plt.legend()
plt.show()

"""

# SEE THE INFLUENCE OF THE REFLECTED TEMPERATURE DEPENDENCE
"""

appReflTemp_C = np.linspace(-30,30,601)
appReflTemp_K = appReflTemp_C + 273.15

reflRawSignal_DN = sensorG/(sensorR*(np.exp(sensorB/(appReflTemp_K))-sensorF)) - sensorO
objRawSignal_DN_ = (rawSignal_DN[125,146]-atmRawSignal_DN*(1-atmTau)-reflRawSignal_DN*(1-objEmissivity)*atmTau)/objEmissivity/atmTau 
temp = sensorB/np.log(sensorG/(sensorR*(objRawSignal_DN_+sensorO))+sensorF)-273.15;

plt.figure(3)
plt.plot(appReflTemp_C, temp)
plt.xlabel("Reflected Temperature [°C]")
plt.ylabel("Corrected Temperature [°C]")
plt.show()
"""

# SEE THE INFLUENCE OF COMBINED REFLECTED TEMPERATURE AND EMISSIVITY
"""
M = 701
appReflTemp_C = np.linspace(-40,30,M)
appReflTemp_K = appReflTemp_C + 273.15
reflRawSignal_DN = sensorG/(sensorR*(np.exp(sensorB/(appReflTemp_K))-sensorF)) - sensorO

N = 701
objEmissivity = np.linspace(0, 1,N)

temp = np.zeros((N,M))

for i, e in enumerate(objEmissivity):
    
    objRawSignal_DN_ = (rawSignal_DN[y_test,x_test]-atmRawSignal_DN*(1-atmTau)-reflRawSignal_DN*(1-e)*atmTau)/e/atmTau 
    temp[i,:] = sensorB/np.log(sensorG/(sensorR*(objRawSignal_DN_+sensorO))+sensorF)-273.15;

temp_min = -50
temp_max = 50
norm = plt.Normalize(vmin=temp_min,vmax=temp_max)

plt.figure(f"d = {objDistance_m}m, $T_atm$ ={airTemp_C} °C, RH ={airRelHumidity_perc} %")
x,y = np.meshgrid(appReflTemp_C, objEmissivity)
plt.imshow(temp, norm = norm,cmap = plt.cm.jet)
#plt.plot(x_test,y_test)
plt.plot(101,694, 'x', color ='red', markersize= 8)
plt.plot(500,694, 'x', color ='red', markersize= 8)
plt.xticks([0,100,200,300,400,500,600,700],['-40','-30','-20','-10','0','10','20','30'])
plt.yticks([0,142,284,426,568,701],['0','0.2','0.4','0.6','0.8','1'])
plt.title(f"DN = {rawSignal_DN[y_test,x_test]}, d = {objDistance_m}m, T_atm ={airTemp_C} °C, RH ={airRelHumidity_perc} %")
plt.ylim(0,701)
plt.xlabel("Reflected Temperature (degC)")
plt.ylabel("Emissivity (-)")

X_VEC = []
Y_VEC = []

for i in range(M):
    for j in range(N):
        pixel = temp[i,j]
        if abs(pixel)<0.05:
            if len(X_VEC)==0 :
                X_VEC.append(j)
                Y_VEC.append(i)
            elif X_VEC[-1] != j:
                X_VEC.append(j)
                Y_VEC.append(i)
       
plt.plot(X_VEC, Y_VEC,color ='black')    
cbar = plt.colorbar()
cbar.set_label("Object Temperature (degC)")

plt.show()

"""
# WATER CONTENT DEPENDENCE
"""
airRelHumidity_perc = np.linspace(0,100,101)
airWaterContent = airRelHumidity_perc /100 * np.exp(1.5587 + 0.06939 * airTemp_C - 0.00027816 * airTemp_C**2 + 0.00000068455 * airTemp_C**3)
atmTau = atmX * np.exp(-np.sqrt(objDistance_m) * (atmAlpha1 + atmBeta1 * np.sqrt(airWaterContent))) + (1-atmX) * np.exp(-np.sqrt(objDistance_m) * (atmAlpha2 + atmBeta2 * np.sqrt(airWaterContent)))

plt.figure(4)
plt.plot(airRelHumidity_perc, atmTau)
plt.xlabel("Air Relative Humidiry")
plt.ylabel("atmTau")
"""

# DN VERSUS TEMPERATURE

def DN_from_T_C(T_C):
    
    DN = sensorG/(sensorR*(np.exp(sensorB/(T_C + 273.15))-sensorF)) - sensorO

    return DN


