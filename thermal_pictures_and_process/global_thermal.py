#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:51:00 2024

@author: antoine
"""

import subprocess
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.close('all')
plt.rcParams['font.family'] = 'Bitstream Vera Sans'
font = 10    
###########################################################################
# SINGLE PART TO MODIFY
# DEFINE PATH AND IR-IMAGE NUMBER

path = "thermal_pictures_and_process/"

scales = [350, 350, 850]
obj_distances = [3500, 4500, 4500]
air_temps = [8,8,10]
x_c = [np.array([173, 409]), np.array([356, 496]), np.array([366, 731])]
y_c = [np.array([333, 323]), np.array([290, 296]), np.array([308, 308])]

for i, IR_IMG in enumerate([2735, 2417, 2587]):                # DC_IMG are n+1
    scale = scales[i]
    scale_interval = 50
    name = "T1"
    # DEFINE USER PARAMETERS
    
    objDistance_m = obj_distances[i]
    airTemp_C = air_temps[i]
    appReflTemp_C = -30
    airRelHumidity_perc = 65    
    objEmissivity = 0.98
    
    x_coords = x_c[i]
    y_coords = y_c[i]
    # DEFINE COORDINATES OF THE PROFILE (OPTIONAL, OR DEFINE THEM LATER)
    
    #Bastille
    #x_coords = np.array([173, 409])
    #y_coords = np.array([333, 323])
    
    
    #Elephant
    #x_coords = np.array([356, 496])
    #y_coords = np.array([290, 296])
    
    #Big
    #x_coords = np.array([366, 731])
    #y_coords = np.array([308, 308])
    
    #Other
    #x_coords = np.array([])
    #y_coords = np.array([])
    
    valh = 40 # height/2 of the plot
    
    # FIND SHIFT PARAMETERS, generally stored in a file
    
    pic_inventory = "thermal_pictures_and_process/IR_inventory.ods"
    df = pd.read_excel(pic_inventory)
    row = df[df['DC'] == IR_IMG + 1]
    
    if not row.empty:
        print("Found")
        approx_shift_x = int(- row['Shiftx'].values[0])
        approx_shift_y = int(710 - row['Shifty'].values[0])
    
    shifts = np.array([approx_shift_x, approx_shift_y])
    
    """
    # Use the mean for a quick process if the shift has not been assessed
    approx_shift_x = -51
    approx_shift_y = 710-648  
    """
    
    ###############################################################################
       
    #==============================================================================
    #========================== FUNCTIONS =========================================
    #==============================================================================
    
    def extract_thermal_image(image_path, output_path):
        """
        Extract the metatdata and the Raw Thermal Image
    
        Parameters
        ----------
        image_path : .jpg
            file from the FLIR camera
        
        output_path: .tif 
            path of the output file
    
        Returns
        -------
        None.
    
        """
        
        command = ["exiftool", "-b", "-RawThermalImage", image_path]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_thermal_image, _ = process.communicate()
    
        if process.returncode != 0:
            print("Error extracting raw thermal image")
            return
        
        with open(output_path, "wb") as file:
            file.write(raw_thermal_image)
        print("Raw thermal image extracted successfully to", output_path)
    
    def extract_metadata(image_path):
        """
        Parameters
        ----------
        image_path : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        """
        cmd = f'exiftool {image_path}'
        res = os.popen(cmd).read()
        metadata = dict()
        for line in res.splitlines():
            l, r = line.split(':', 1)
            metadata[l.strip()] = r
        
        return metadata
    
    def compute_temperature(ar_t, u_params, c_params):
        
        """
        This function uses the IRimage algorithm to convert a raw sensed DN in an objet surface temperature.
    
        Parameters
        ----------
        img : numpy.ndarray
            DN values from the camera
        u_params : dict
            user-selected parameters
        c_params : dict
            camera calibration parameter
            
        Returns
        -------
        objTemp_C : numpy.ndarray
            Temperature values
    
        """
        
        # First, retrieved parameters
        
        appReflTemp_C = u_params.get('appReflTemp_C') 
        objEmissivity = u_params.get('objEmissivity') 
        airTemp_C = u_params.get('airTemp_C') 
        airRelHumidity_perc = u_params.get('airRelHumidity_perc') 
        objDistance_m = u_params.get('objDistance_m')
        
        atmAlpha1 = c_params.get('atmAlpha1')
        atmAlpha2 = c_params.get('atmAlpha2')
        atmBeta1 = c_params.get('atmBeta1')
        atmBeta2 = c_params.get('atmBeta2')
        atmX = c_params.get('atmX')
        
        sensorG = c_params.get('sensorG')
        sensorB = c_params.get('sensorB')
        sensorF = c_params.get('sensorF')
        sensorO = c_params.get('sensorO')
        sensorR = c_params.get('sensorR')
        
        
        # CALCULATE DERIVED VARIABLES
    
        appReflTemp_K = appReflTemp_C + 273.15
        airTemp_K = airTemp_C + 273.15
        airWaterContent = airRelHumidity_perc /100 * np.exp(1.5587 + 0.06939 * airTemp_C - 0.00027816 * airTemp_C**2 + 0.000000068455 * airTemp_C**3)
        atmTau = atmX * np.exp(-np.sqrt(objDistance_m) * (atmAlpha1 + atmBeta1 * np.sqrt(airWaterContent))) + (1-atmX) * np.exp(-np.sqrt(objDistance_m) * (atmAlpha2 + atmBeta2 * np.sqrt(airWaterContent)))
        atmRawSignal_DN = sensorG/(sensorR*(np.exp(sensorB/(airTemp_K))-sensorF)) - sensorO
        reflRawSignal_DN = sensorG/(sensorR*(np.exp(sensorB/(appReflTemp_K))-sensorF)) - sensorO
    
        # CALCULATE TEMPERATURE VALUES FOR EACH PIXEL
    
        rawSignal_DN = ar_t
        objRawSignal_DN = (rawSignal_DN-atmRawSignal_DN*(1-atmTau)-reflRawSignal_DN*(1-objEmissivity)*atmTau)/objEmissivity/atmTau
        objTemp_C = sensorB/np.log(sensorG/(sensorR*(objRawSignal_DN+sensorO))+sensorF)-273.15
        
        return objTemp_C
    
    def array_to_jpeg(array, jpeg_path):
        # Normalize the array to 0-255 range
        normalized_array = ((array - np.min(array)) / (np.max(array) - np.min(array)) * 255).astype(np.uint8)
    
        # Convert the normalized array to a PIL Image
        image = Image.fromarray(normalized_array)
    
        # Convert the image to 'RGB' mode if it's in 'F' mode (floating-point)
        if image.mode == 'F':
            image = image.convert('RGB')
    
        # Save as JPEG
        image.save(jpeg_path, "JPEG")
    
    def choose_profile_points(visible_aligned, approx_shift_x, approx_shift_y):
        
        # Display the image
        plt.figure(1)
        plt.imshow(visible_aligned)
        #plt.plot([-approx_shift_x, objColorRGB.shape[1], objColorRGB.shape[1], -approx_shift_x, -approx_shift_x],
        #         [0, 0, objColorRGB.shape[0] - approx_shift_y, objColorRGB.shape[0] - approx_shift_y, 0], color='red')
        plt.title("Draw a Line, DC_" + str(IR_IMG + 1))
        
        x_coords = []
        y_coords = []
    
        
        # Initialize a list to store lines
        lines = []
        
        # Function to update the plot with the polyline
        
        def update_plot():
            for i in range(len(lines)):
                lines[i].set_data(x_coords[i:i + 2], y_coords[i:i + 2])
    
            plt.gca().figure.canvas.draw()
    
        # Function to handle mouse click events
        def onclick(event):
            nonlocal x_coords, y_coords  # Use nonlocal to access outer scope variables
    
            if event.button == 1:  # Left click
                x_coords.append(event.xdata)
                y_coords.append(event.ydata)
    
                # Update the plot with the new line segment
                if len(x_coords) >= 2:
                    line = Line2D(x_coords[-2:], y_coords[-2:], color='red')
                    plt.gca().add_line(line)
                    lines.append(line)
                    update_plot()
            elif event.button == 3:  # Right click
                # If there are at least two points, finish the line
                if len(x_coords) >= 2:
                    # Disconnect the event handler to stop further input
                    plt.gcf().canvas.mpl_disconnect(cid)
                    #print("Final coordinates:", x_coords, y_coords)
                    return x_coords  # Return x_coords when right-click occurs
                else:
                    plt.gcf().canvas.mpl_disconnect(cid)  # Disconnect even if there are no coordinates
                    print("No coordinates selected.")
                    return None
            # Returning `True` consumes the event, preventing further handling
            return True
        
        # Connect the onclick function to the mouse click event
        cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    
        return x_coords, y_coords
    
    def find_shift_using_hough(DC_jpg_path, IR_jpg_path):
        
        visible_img = cv2.imread(DC_jpg_path)
        height_v = np.shape(visible_img)[0]
        width_v = np.shape(visible_img)[1]
        
        thermal_img = cv2.imread(IR_jpg_path)
        thermal_img = cv2.resize(thermal_img, (width_v, height_v), interpolation=cv2.INTER_AREA)
        # Convert images to grayscale
        gray1 = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(visible_img, cv2.COLOR_BGR2GRAY)
    
        # Apply Canny edge detection
        edges1 = cv2.Canny(gray1, 50, 150, apertureSize=3)
        edges2 = cv2.Canny(gray2, 50, 150, apertureSize=3)
    
        # Apply Hough Transform to detect lines
        lines1 = cv2.HoughLines(edges1, 1, np.pi / 180, 100)
        lines2 = cv2.HoughLines(edges2, 1, np.pi / 180, 100)
    
        # Calculate the angle and distance of the lines
        angle1 = np.mean([line[0][1] for line in lines1])
        angle2 = np.mean([line[0][1] for line in lines2])
        distance1 = np.mean([line[0][0] for line in lines1])
        distance2 = np.mean([line[0][0] for line in lines2])
    
        # Calculate the shift based on the difference in angle and distance
        angle_diff = np.abs(angle1 - angle2)
        distance_diff = np.abs(distance1 - distance2)
        shift_x = distance_diff * np.cos(angle_diff)
        shift_y = distance_diff * np.sin(angle_diff)
        
        # Draw detected lines on images
        draw_lines(visible_img, lines1)
        draw_lines(thermal_img, lines2)
    
        return shift_x, shift_y, visible_img, thermal_img
    
    def draw_lines(image, lines):
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    
    def calculate_shift(DC_jpg_path, IR_jpg_path):
        
        # Convert images to grayscale
        
        visible_img = cv2.imread(DC_jpg_path)
        height_v = np.shape(visible_img)[0]
        width_v = np.shape(visible_img)[1]
        
        thermal_img = cv2.imread(IR_jpg_path)
        thermal_img = cv2.resize(thermal_img, (width_v, height_v), interpolation=cv2.INTER_AREA)
        
        visible_gray = visible_img #cv2.cvtColor(visible_img, cv2.COLOR_BGR2GRAY)
        thermal_gray = thermal_img
        print("start akaze")
        akaze = cv2.AKAZE_create()
    
        # Find keypoints and descriptors
        kp_visible, des_visible = akaze.detectAndCompute(visible_gray, None)
        kp_thermal, des_thermal = akaze.detectAndCompute(thermal_gray, None)
    
        # Match keypoints
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des_visible, des_thermal, k=2)
    
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        print("matches dones")
        # Plot matches
        combined_image = np.zeros((height_v, 2 * width_v, 3), dtype=np.uint8)
        combined_image[:, :width_v] = visible_img
        combined_image[:, width_v:] = thermal_img
    
        for match_ in good_matches:
            pt1 = kp_visible[match_.queryIdx].pt
            pt2 = kp_thermal[match_.trainIdx].pt
            pt2 = (int(pt2[0]) + width_v, int(pt2[1]))
    
            cv2.circle(combined_image, (int(pt1[0]), int(pt1[1])), 5, (0, 255, 0), 2)
            cv2.circle(combined_image, (int(pt2[0]), int(pt2[1])), 5, (0, 255, 0), 2)
            cv2.line(combined_image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 0), 2)
    
        cv2.imshow("Matched image", combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        # Calculate shift
        shift_x = [kp_thermal[match.trainIdx].pt[0] - kp_visible[match.queryIdx].pt[0] for match in good_matches]
        shift_y = [kp_thermal[match.trainIdx].pt[1] - kp_visible[match.queryIdx].pt[1] for match in good_matches]
    
        # Calculate median shift
        median_shift_x = int(round(np.median(shift_x)))
        median_shift_y = int(round(np.median(shift_y)))
    
        return shift_x, shift_y, int(round(median_shift_x)), int(round(median_shift_y))
    
    
    def blue_profile(x_coords, y_coords, visible_aligned):
        
        # This works with the REFERENCE visible image
        
        x1_v = round(x_coords[0])  
        x2_v = round(x_coords[1])
        y1_v = y_coords[0] 
        y2_v = y_coords[1]
        
        min_xv = min(x1_v, x2_v)
        max_xv = max(x1_v, x2_v)
        
        print(x2_v-x1_v)
        print(y2_v-y1_v)
        
        if abs(x2_v-x1_v) > abs(y2_v-y1_v):
        
            p_xv = np.linspace(x1_v, x2_v, x2_v-x1_v + 1)
            slope = (y2_v-y1_v)/(x2_v-x1_v)
            ind_yv = y1_v + slope*(p_xv - x1_v)
            p_yv = np.round(ind_yv)
            
            p_xv = np.int64(p_xv)
            p_yv = np.int64(p_yv)
            plt.plot(p_xv,p_yv)
        
        red = visible_aligned[p_yv, p_xv, 0]
        green = visible_aligned[p_yv, p_xv, 1]
        blue = visible_aligned[p_yv, p_xv, 2]
        
        blue = blue.astype(np.float64)
        green = green.astype(np.float64)
        red = red.astype(np.float64)
        
        return red, green, blue, p_xv, p_yv
        
    
    def temperature_profile(x_coords, y_coords, thermal_aligned):
    
        # This works with the SHIFTED (NOT ANYMORE!!!) thermal image
        
        x1_t = round(x_coords[0]) # not needed anymore since aligned + approx_shift_x)  
        x2_t = round(x_coords[1]) # + approx_shift_x)
        y1_t = y_coords[0]# + approx_shift_y
        y2_t = y_coords[1]# + approx_shift_y
        
        min_xt = min(x1_t, x2_t)
        max_xt = max(x1_t, x2_t)
        
        if abs(x2_t-x1_t) > abs(y2_t-y1_t):
        
            p_xt = np.linspace(x1_t, x2_t, x2_t-x1_t + 1)
            slope = (y2_t-y1_t)/(x2_t-x1_t)
            ind_yt = y1_t + slope*(p_xt - x1_t)
            p_yt = np.round(ind_yt)
            
            p_xt = np.int64(p_xt)
            p_yt = np.int64(p_yt)
            plt.plot(p_xt,p_yt)
        
        temp = thermal_aligned[p_yt, p_xt]
        
        return temp, p_xt, p_yt
    
    
    #============================ " MAIN " ========================================
    
     # Convert the user parameters in a dictionary
     
    u_params = {
        'appReflTemp_C': appReflTemp_C, 
        'objEmissivity': objEmissivity, 
        'airTemp_C': airTemp_C, 
        'airRelHumidity_perc': airRelHumidity_perc, 
        'objDistance_m': objDistance_m     }
      
    IR_image_path = f"{path}IR_{IR_IMG}.jpg"
    DC_image_path = f"{path}DC_{IR_IMG + 1}.jpg"
    output_path = f"{path}IR_{IR_IMG}.tif"
     
    # Extract thermal image from the FLIR output and the metadata
     
    extract_thermal_image(IR_image_path, output_path)
    metadata = extract_metadata(IR_image_path)
     
    # compute the shift between the thermal and visible image (do it when it is drawn on the profile with a smaller rectangle (better accuracy!!))
    #shift_x, shift_y, med_shift_x, med_shift_y = calculate_shift(DC_image_path, output_path) does not work for the moment
     
    # Extrat camera parameters from the metadata and stack them into a dictionary
     
    atmAlpha1 = float(metadata["Atmospheric Trans Alpha 1"])
    atmAlpha2 = float(metadata["Atmospheric Trans Alpha 2"])
    atmBeta1 = float(metadata["Atmospheric Trans Beta 1"])
    atmBeta2 = float(metadata["Atmospheric Trans Beta 2"])
    atmX = float(metadata["Atmospheric Trans X"])
    
    sensorG = float(metadata["Planck R1"])
    sensorB = float(metadata["Planck B"])
    sensorF = float(metadata["Planck F"])
    sensorO = float(metadata["Planck O"])
    sensorR = float(metadata["Planck R2"])
     
    c_params = {
        'atmAlpha1': atmAlpha1,
        'atmAlpha2': atmAlpha2,
        'atmBeta1': atmBeta1,
        'atmBeta2': atmBeta2,
        'atmX': atmX,
        'sensorG': sensorG,
        'sensorB': sensorB,
        'sensorF': sensorF,
        'sensorO': sensorO,
        'sensorR': sensorR
        }
     
    # Get the thermal image img
     
    im_t = Image.open(output_path)
    ar_t = np.array(im_t)
    height_t, width_t = np.shape(ar_t)
    # Get the visible image im_v
    
    # check if there is a lighter picture:
    
    im_v = Image.open(DC_image_path)
        
    objColorRGB = np.array(im_v)
    height_v, width_v, channels_v = np.shape(objColorRGB)
     
    # Compute the temperature and save it to a jpeg file
    
    objTemp_C = compute_temperature(ar_t, u_params, c_params)
    
    """
    jpeg_file = f"{path}objTemp_C_{IR_IMG}.jpg"
    array_to_jpeg(objTemp_C, jpeg_file)
    """
     
    # Resize the thermal image to get the same resolution as the visible
    
    scaled_objTemp_C = cv2.resize(objTemp_C, (width_v, height_v), interpolation=cv2.INTER_CUBIC)
    scaled_objTemp_C_no_interp = cv2.resize(objTemp_C, (width_v, height_v), interpolation=None)
    
    # Choose profile points (possible to draw a multiline, but first only start with 2 segments)
    # Here, the approx_shift_x/y is replaced by the value found manually while superimposing icebergs
    
    
    thermal_aligned = scaled_objTemp_C[approx_shift_y:, :approx_shift_x] # is SCALED WITH CUBIC INTERPOLATION
    visible_aligned = objColorRGB[:-approx_shift_y, -approx_shift_x:]
    
    scale_ind = "yes"
    
    """
    # Interactive manual choose of the coordinates
    x_input = input("X_coords: format = x1 x2  : ")
    y_input = input("Y_coords: format = y1 y2  : ")
    
    x_coords = np.array(x_input.split(), dtype=float)
    y_coords = np.array(y_input.split(), dtype=float)
    """
    
    if x_coords.size == 0 or y_coords.size == 0:
        scale_ind = "yes"
        x_coords, y_coords = choose_profile_points(visible_aligned, approx_shift_x, approx_shift_y)
    
    
    while len(x_coords) < 2: # To change later to adapt to longer lines
        plt.pause(0.1)
        
    # Align the two images focussing on the extent of the line
    # This part has to be done manually yet
    
    
    lv_x = np.min(x_coords)        # left visible x
    rv_x = np.max(x_coords)        # right x
    uv_y = np.max(y_coords)        # upper y, WARNING, y goes from top to bottom   
    lv_y = np.min(y_coords)        # lower y
    
    """
    #shift_x, shift_y, me_shift_x, med_shift_y = calculate_shift(DC_image_path, jpeg_file)
    #shift_x, shift_y, visible_img, thermal_img = find_shift_using_hough(DC_image_path, jpeg_file)
    """
    
    # Align first the two rasters, then the indexation will be the same for both 
    
    
    red, green, blue, p_xv, p_yv = blue_profile(x_coords, y_coords, visible_aligned)  
    temp, p_xt, p_yt = temperature_profile(x_coords, y_coords, thermal_aligned)
    
    z1 = blue/(blue+green+red)
    z2 = blue/red
    z3 = blue/green
    z4 = (blue-red)/(blue+red)
    
    """
    ratio = np.zeros(len(red))
    for i in range(len(red)):
        ratio[i] = blue[i] / (red[i] + green[i] + blue[i])
        if red[i] + green[i] + blue[i] == 0:
            print(red[i])
            print(blue[i])
            print(green[i])
    """
    
    #==============================================================================
    #=============================== PLOTS :D =====================================
    #==============================================================================
    
    fig, ax = plt.subplots(3,1, gridspec_kw={'height_ratios': [1, 1, 1]})
    
    # Visible zoom
        
    ax0 = ax[0]
    ax0.set_title(name, fontweight='bold', fontsize = font)
    ax0.imshow(visible_aligned) 
    mp_yv = (max(p_yv) + max(p_yv))/2
    ax0.set_ylim((mp_yv + valh, mp_yv - valh))
    ax0.plot([p_xv[0], p_xv[-1]],[ p_yv[0], p_yv[-1]], color = 'black')
    ax0.set_xlim((p_xv[0], p_xv[-1]))
    ax0.set_xticks([],[])
    ax0.set_yticks([],[])
    
    
    
    #ax0.set_xlabel("Pixel coordinates")
    
    pix_sca = (len(p_xv))/scale
    incl = np.arctan((y_coords[1] - y_coords[0])/(x_coords[1]-x_coords[0]))
    ti = 5 
    
    if scale_ind == 'yes':
        ax0.plot(x_coords, y_coords, color = 'black')
        
        line_end = [x_coords[1], y_coords[1]]
        line_start = [x_coords[0], y_coords[1]]
        line_length = np.linalg.norm(np.array(line_end) - np.array(line_start))
        
         
        vector_xti = [0]
        vector_x = [0]
        n = scale//scale_interval
        for j in range(1, n+1):
            vector_xti.append(j*scale_interval*pix_sca)
            vector_x.append(j)
        
        for i in vector_xti:
            i = int(round(i))
            #rect = Rectangle((line_start[0] + i, line_start[1] - 5), scale_length, 10, color='blue', alpha=0.5)
            #ax4.add_patch(rect)
            x_cm = line_start[0] + i
            y_cm = np.tan(incl)*(i) + y_coords[0]
            ax0.plot([x_cm - ti*np.tan(incl), x_cm + ti*np.tan(incl)], [y_cm + np.cos(incl)*ti, y_cm - np.cos(incl)*ti], color='black')
            
            if i == 0:
                ax0.text(line_start[0] + scale_interval*pix_sca/2, y_coords[0] + 10 ,"50 m", fontweight = 'bold', color='black', ha = 'center', rotation = incl)
    
    
    
    # Thermal profile
    
    ax2 = ax[1]
    ax2.plot(p_xt,temp, color = 'red', label = "Temp")
    ax2.set_xlim(p_xv[0], p_xv[-1])
    ax2.set_ylim((-9,1))
    ax2.legend(loc = "upper right")
    
    ax2b = ax2.twinx()
    ax2b.plot(p_xt, z1, color = 'blue', label = "b/(r+g+b)")
    ax2.set_ylabel("Temperature (degC)")
    ax2b.set_ylabel("Blue index")
    ax2b.legend(loc = "upper left")
    
    #ax2b.plot(p_xt, z2, color = 'violet', label = "b/r")
    #ax2b.plot(p_xt, z3, color = 'green', label = "b/g")
    #ax2b.plot(p_xt, z4, color = 'black', label = "b-r/b+r")
    # Display the scale in meters
    
    xticks = []
    xticks_lab = []
    
    
    for i, item in enumerate(vector_xti):
        xticks.append(p_xv[0] + item)    
        xticks_lab.append(str(vector_x[i]*50))
        
    ax2.set_xticks(xticks, xticks_lab, fontsize = font)
    ax2.set_xlabel("Distance in meters")
    ax2.legend()
    ax2b.legend()
    
    
    # Thermal image zoom
    
    ax1 = ax[2]
    img = ax1.imshow(thermal_aligned, cmap = plt.cm.inferno, vmin=-10, vmax=1)
    #cbar = plt.colorbar(img,  orientation='horizontal', shrink = 2)
    #cbar.ax.tick_params(labelsize=15)  
    ax1.plot([p_xt[0], p_xt[-1]], [p_yt[0], p_yt[-1]], color='violet')
    ax1.set_xlim((p_xv[0], p_xv[-1]))
    ax1.set_ylim((mp_yv + valh, mp_yv - valh))
    ax1.set_xticks([],[])
    ax1.set_yticks([],[])
    par = f"d={objDistance_m} m, e={objEmissivity}, RH={airRelHumidity_perc} %, Tatm={airTemp_C} °C, Trefl={appReflTemp_C} °C"
    ax1.set_xlabel(f"d={objDistance_m} m, e={objEmissivity}, RH={airRelHumidity_perc} %, Tatm={airTemp_C} °C, Trefl={appReflTemp_C} °C")
    pix_sca = (len(p_xv)-1)/scale
    
    
    
    if scale_ind == 'yes':
        ax1.plot(x_coords, y_coords, color = 'black')
        
        line_end = [x_coords[1], y_coords[1]]
        line_start = [x_coords[0], y_coords[1]]
        line_length = np.linalg.norm(np.array(line_end) - np.array(line_start))
        
        vector_xti = []
        n = scale//scale_interval
        for j in range(n):
            vector_xti.append(j*scale_interval*pix_sca)
        
        for i in vector_xti:
            i = int(round(i))
            scale_length = min(scale_interval, int(line_length) - i)
            x_cm = line_start[0] + i
            y_cm = np.tan(incl)*(i) + y_coords[0]
            ax1.plot([x_cm - ti*np.tan(incl), x_cm + ti*np.tan(incl)], [y_cm + np.cos(incl)*ti, y_cm - np.cos(incl)*ti], color='black')
            
            if i == 0:
                ax1.text(line_start[0] + scale_interval*pix_sca/2, y_coords[0] + 10 ,"50 m", fontweight = 'bold', color='black', ha = 'center', rotation = incl)
                
    for ax_ in ax:
        ax_.set_anchor('W')
    
    plt.tight_layout()
    
    save_path = "thermal_pictures_and_process/"
    np.save(f"{save_path}{IR_IMG}_pxt", p_xt)
    np.save(f"{save_path}{IR_IMG}_pxv", p_xv)
    np.save(f"{save_path}{IR_IMG}_pyv", p_yv)
    np.save(f"{save_path}{IR_IMG}_p_yt", p_yt)
    np.save(f"{save_path}{IR_IMG}_temp", temp)
    np.save(f"{save_path}{IR_IMG}_z1", z1)
    np.save(f"{save_path}{IR_IMG}_vis", visible_aligned)
    np.save(f"{save_path}{IR_IMG}_the", thermal_aligned)
    np.save(f"{save_path}{IR_IMG}_par", par)
    np.save(f"{save_path}{IR_IMG}_shifts", shifts)
    np.save(f"{save_path}{IR_IMG}_xcoords", x_coords)
    np.save(f"{save_path}{IR_IMG}_ycoords", y_coords)
    np.save(f"{save_path}{IR_IMG}_scale", scale)
    np.save(f"{save_path}{IR_IMG}_scaleint", scale_interval)
    np.save(f"{save_path}{IR_IMG}_vector_xti", vector_xti)
    
    
    #==============================================================================
    # Old way of plotting (2x2)
    """
    fig, ax = plt.subplots(2,2)#, gridspec_kw={'height_ratios': [1, 2]})
    
    # Thermal image
    ax1 = ax[0,0]
    img = ax1.imshow(thermal_aligned, cmap = plt.cm.inferno, vmin=-3, vmax=1)
    ax1.plot([p_xt[0], p_xt[-1]], [p_yt[0], p_yt[-1]], color='red')
    cbar = plt.colorbar(img)
    
    # Thermal profile
    ax2 = ax[0,1]
    #ax2.imshow(visible_aligned)
    #ax2.plot([p_xv[0], p_xv[-1]],[ p_yv[0], p_yv[-1]], color = 'black')
    ax2.plot(p_xt,temp, color = 'red', label = "Temp")
    ax2b = ax2.twinx()
    ax2b.plot(p_xt, z1, color = 'blue', label = "b/(r+g+b)")
    #ax2b.plot(p_xt, z2, color = 'violet', label = "b/r")
    #ax2b.plot(p_xt, z3, color = 'green', label = "b/g")
    #ax2b.plot(p_xt, z4, color = 'black', label = "b-r/b+r")
    ax2.set_xlim(p_xv[0], p_xv[-1])
    
    # Display the scale in meters
    xticks = []
    xticks_lab = []
    pix_sca = (len(p_xv))/scale
    
    for i, item in enumerate(p_xv):
        if i%round(scale_interval*pix_sca) == 0:
            xticks.append(item)
            xticks_lab.append(str(int((item-p_xv[0])/pix_sca)))
            
    
    ax2.set_xticks(xticks, xticks_lab)
    ax2.set_xlabel("Distance in meters")
    ax2.legend()
    ax2b.legend()
    
    # Visible image
    ax3 = ax[1,0]
    ax3.imshow(visible_aligned)
    ax3.plot([p_xv[0], p_xv[-1]], [p_yv[0], p_yv[-1]], color='red')
    ax3.set_xlabel(f"d={objDistance_m}, e={objEmissivity}, RH={airRelHumidity_perc}, Tatm={airTemp_C}, Trefl={appReflTemp_C}")
    
    # Zoom on the visible image¨
    ax4 = ax[1,1]
    
    ax4.imshow(visible_aligned) 
    ax4.plot([p_xv[0], p_xv[-1]],[ p_yv[0], p_yv[-1]], color = 'black')
    ax4.set_xlim((p_xv[0], p_xv[-1]))
    ax4.set_ylim((max(p_yv) + 20), min(p_yv) - 20)
    ax4.set_xlabel("Pixel coordinates")
    # Plot the scale
    if scale_ind == 'yes':
        ax4.plot(x_coords, y_coords, color = 'violet')
        
        line_end = [x_coords[1], y_coords[1]]
        line_start = [x_coords[0], y_coords[1]]
        line_length = np.linalg.norm(np.array(line_end) - np.array(line_start))
        
            
        for i in range(0, int(line_length), int(scale_interval*pix_sca)):
            scale_length = min(scale_interval, int(line_length) - i)
            #rect = Rectangle((line_start[0] + i, line_start[1] - 5), scale_length, 10, color='blue', alpha=0.5)
            #ax4.add_patch(rect)
            ax4.plot([line_start[0] + i, line_start[0] + i], [line_start[1] - 5, line_start[1] + 5], color='black')
            
            if i == 0:
                ax4.text(line_start[0] + scale_interval*pix_sca/2, line_start[1] + 10 ,"50 m", color='black', ha = 'center')
    
    #ax4.axis('equal')
    """
    # Plot thermal
    """
    fig1, ax1 = plt.subplots()
    ax1.imshow(objTemp_C, cmap = plt.cm.inferno)
    
    # Plot visible
    fig2, ax2 = plt.subplots()
    ax2.imshow(im_v)
    """
    
    """
    Forget the shift for the moment
    shift_x, shift_y = calculate_shift(DC_image_path, IR_image_path)
    print("Shift in X:", shift_x, "pixels")
    print("Shift in Y:", shift_y, "pixels")
    """

