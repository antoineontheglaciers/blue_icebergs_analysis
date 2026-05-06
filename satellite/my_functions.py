#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:13:10 2024

@author: antoine
"""

import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
import numpy as np
import os, glob
from array import array
import csv
import pandas as pd
import struct
import re
from numpy import asarray
from numpy import savetxt
import matplotlib
import math
import time
from  datetime import datetime
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as path_effects
import geopandas
from astropy.coordinates import get_sun, AltAz, EarthLocation
from astropy.time import Time
import astropy.units as u
import pvlib
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, mapping

# =============================================================================
# Function that links the band number with the index for the storage in python array

def dumb_but_effective(var_name):
    
    global index
    
    # For both satellites
    if var_name == 'B01':
        index = 0
    elif var_name == 'B02':
        index = 1
    elif var_name == 'B03':
        index = 2
    elif var_name == 'B04':
        index = 3
    elif var_name == 'B05':
        index = 4
    elif var_name == 'B06':
        index = 5
    elif var_name == 'B07':
        index = 6
    elif var_name == 'B08':
        index = 7
    elif var_name == 'B8A':
        index = 8
    elif var_name == 'B09':
        index = 9
    elif var_name == 'B11':
        index = 10
    elif var_name == 'B12':
        index = 11
    else:
        raise ValueError("There is no clear band name, look at the filename")
        
    return index

# =============================================================================
# Read tiff files and store the data

def read_and_store(path, shp_path, date):  # FOR GENERAL SIGNATURE
    
    shapefile = gpd.read_file(shp_path)
    selected_feature = shapefile[shapefile['date'] == date]
    
    if len(selected_feature) == 1:
        union_geometry = selected_feature.geometry.values[0]
    else:
        # Create a single geometry representing the union of all selected geometries
        union_geometry = selected_feature.geometry.unary_union
        
    print(f"Geometry: {union_geometry}\n")

    # In a given fold (e.g. sea_2023_07_02), find all the tiff files
    print(f"PATH: {path}\n")
    picList = glob.glob(os.path.join(path,"*.tif*"))
    print(f"PicList: {picList}\n")

    # remove the pictures that are not single bands (e.g. Thermal, Optimized color, Classification,...)

    singleBands = []
    
    for i in range(len(picList)):
        
        if picList[i].find("olor") == -1 and picList[i].find("Thermal") == -1 and picList[i].find("classification") == -1 and picList[i].find("SWIR") == -1 and picList[i].find("xml") == -1:
            singleBands.append(picList[i])
   
    n_bands = len(singleBands)
    #print(newpicList)        
    
    print(f"n_bands: {n_bands}\n")
    
    maxrast = np.zeros(n_bands)
    minrast = np.zeros(n_bands)
    mean_val = np.zeros(n_bands)
    median_val = np.zeros(n_bands)
    std_val = np.zeros(n_bands)
    
    singleBands.sort()
    print(f"Single bands : {singleBands}\n")
    
    print("START PROCESSING -----------------------------------------")
    for i, bandname in enumerate(singleBands):
        
        # To find which band it is (B01, B02, etc.), we want to find the indicator
        # in the bandname "_Bxx"
        
        _B = "_B"
        _Bfound = bandname.rfind(_B)
        
        if _Bfound == -1:
            raise ValueError(f"No band name indicator {_B} was found")
        else:
            var_name = bandname[_Bfound + 1:_Bfound + 4]
            print(f"{bandname} -->\n {var_name}")
            
        index = dumb_but_effective(var_name)
         
        print(f"index: {index}\n")
            
        # Open the raster, read it and clean data (convert values out of the shp to 'nan')
        """
        data = rasterio.open(bandname)
        rast = data.read(1)
        rast[rast == -1] = float('nan')
        """
        
        #print(f"------------\n {bandname} is processed\n data: {data}")
        
        # Compute statistics of the raster
        
        with rasterio.open(bandname) as src:
            # Mask the raster with the shapefile's geometry
            rast, transform = mask(src, [union_geometry], nodata = float('Nan'), crop=True)
            # Max and min
        
            maxrast[index] = np.nanmax(rast)
            minrast[index] = np.nanmin(rast)
        
            #print(f"Max {var_name}: {maxrast}")
            #print(f"Min {var_name}: {minrast}")
              
            # Mean, median, std
        
            mean_val[index] = np.nanmean(rast)
            median_val[index] = np.nanmedian(rast)
            std_val[index] = np.nanstd(rast)
        

        # Get the transformation (useful for georeferenced plot)
        # TO SOLVE
        # bounds_dc[feat_date] = data.bounds
        print('END PROCESSING OF THE BAND\n')
    print("END PROCESSING OF THE BANDS --------------------------------------\n")
    
    """
    print(f"maxrast: {maxrast}\n")    
    print(f"minrast: {minrast}\n")
    print(f"mean_val: {mean_val}\n")
    print(f"median_val: {median_val}\n")
    print(f"std_val: {std_val}\n")
    """
    
    return maxrast, minrast, mean_val, median_val, std_val

# =============================================================================
# Clip the raster with a given item in a shapefile

def clip_the_raster(shapefile_path, date, raster_path,output_path):
    
    # Read shapefile and filter by the desired field
    shapefile = gpd.read_file(shapefile_path)
    print(shapefile)
    selected_feature = shapefile[shapefile['date'] == date]

        
    print(f"selected_feature: {selected_feature}")
    
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Extract the geometry of the selected feature
        geometry = selected_feature.geometry.values[0]
    
        geometry = shape(geometry).buffer(0).envelope
    
        # Convert the geometry to GeoJSON format
        geojson_geometry = mapping(geometry)
    
        # Clip the raster using the shapefile's geometry
        clipped_raster, transform = mask(src, [geojson_geometry], crop=True, nodata=-1)
    
        # Update metadata for the clipped raster
        meta = src.meta.copy()
        meta.update({
            'driver': 'GTiff',
            'height': clipped_raster.shape[1],
            'width': clipped_raster.shape[2],
            'transform': transform,
            'nodata':-1
        })
    
        # Write the clipped raster to the specified output path
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(clipped_raster)

# =============================================================================

# Read tiff files and store the data

def read_and_store_evolution(path, date): # FOR EVOLUTION 
    
    # In a given fold (e.g. sea_2023_07_02), find all the tiff files
    print("Path:", path)
    picList = glob.glob(os.path.join(path,"*.tif*"))
    #print("picList:", picList)

    # remove the pictures that are not single bands (e.g. Thermal, Optimized color, Classification,...)

    singleBands = []
    
    for i in range(len(picList)):
        
        print("picList[i]:", picList[i])
        print("type(picList[i])", type(picList[i]))
        if picList[i].rfind("olor") == -1 and picList[i].rfind("Thermal") == -1 and picList[i].rfind("classification") == -1 and picList[i].rfind("SWIR") == -1 and picList[i].rfind("shp") == -1 and picList[i].rfind("xml") == -1:
            singleBands.append(picList[i])
   
    n_bands = len(singleBands)
    #print(newpicList)        
    
    maxrast = np.zeros(n_bands)
    minrast = np.zeros(n_bands)
    mean_val = np.zeros(n_bands)
    median_val = np.zeros(n_bands)
    std_val = np.zeros(n_bands)
    
    singleBands.sort()
    print("singleBands:", singleBands)
    print("Maxrast:", maxrast)
    print("N bands:", n_bands)
    
    print("START PROCESSING -----------------------------------------")
    
    for k, bandname in enumerate(singleBands):
        
        # To find which band it is (B01, B02, etc.), we want to find the indicator
        # in the bandname "_Bxx"
        
        _B = "_B"
        _Bfound = bandname.rfind(_B)
        
        if _Bfound == -1:
            raise ValueError(f"No band name indicator {_B} was found")
        else:
            var_name = bandname[_Bfound + 1:_Bfound + 4]
            print(f"{bandname} -->\n {var_name}")
            
        index = dumb_but_effective(var_name)
            
            
        # Open the raster, read it and clean data (convert values out of the shp to 'nan')
        """
        data = rasterio.open(bandname)
        rast = data.read(1)
        rast[rast == -1] = float('nan')
        """
        
        #print(f"------------\n {bandname} is processed\n data: {data}")
        
        # Compute statistics of the raster
        
        data = rasterio.open(bandname)
        rast = data.read(1)
        rast[rast == -1] = float("nan")
        
        mean_val[index] = np.nanmean(rast)
        median_val[index] = np.nanmedian(rast)
        std_val[index] = np.nanstd(rast)
        
        maxrast[index] = np.nanmax(rast)
        minrast[index] = np.nanmin(rast)
        
        print(f"Max {var_name}: {maxrast}")
        print(f"Min {var_name}: {minrast}")
              
        # Mean, median, std
        

        

        # Get the transformation (useful for georeferenced plot)
        # TO SOLVE
        # bounds_dc[feat_date] = data.bounds
        print('END PROCESSING')

    return maxrast, minrast, mean_val, median_val, std_val