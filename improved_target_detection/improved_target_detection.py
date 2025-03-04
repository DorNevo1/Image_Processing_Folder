# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 16:54:22 2025

@author: dorne
"""


### this example code temaplate uses SPy spectral python library implementation of ACE and Matched Filter detectors
### to test you own target detection algorithms:
    ### modify the following functions: 
        ### GetGroundTruthSpectra: to appropriately select target pixels spectra from ground truth and input HSI images
        ### ApplyDetector: implement your own detector algorithm:
            # the argument truth_spectra come from the function GetGroundTruthSpectra
            # the argument input_img is the HSI image where target detection is to be run
            # the argument mask_img is the binary image where zero pixels indicate locations to ignore in target detection
    ### in the main function modify appropraite paths for the following:
        # input_img: path where the HSI images are (or/and .hdr files)
        # mask_img = location of the the binary mask image (N.B. you can create your own mask image)
        # truth_img_green = location of the ground truth image of the 66 target green pixels
        # truth_img_gray = location of the ground truth image of the 60 target gray pixels 

#### spectral python (Spy): documentation: https://www.spectralpython.net/index.html

import numpy as np
import spectral as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import auc
from scipy.ndimage import uniform_filter
###############################################################################
###############################################################################
###############################################################################

### return a list of ground truth pixels for a target class
### modify this function to extract target pixel spectra using the gorund truth/classification image and input HSI image




def GetGroundTruthSpectra(input_img, gt_mask, algo):
    
    print("Getting ground truth spectra...")
    
    truth_arr = np.copy(np.asarray(gt_mask.load()))

    #print(truth_arr.shape)


    truth_list = tuple(zip(*(np.where(truth_arr == 1))))
    
    print("Ground Truth Pixels: ", len(truth_list))

    ### we can pass multiple target spectra to ACE in SPy implementation. Here we take 40
    ### when using all the target pixel spectra as target, the ACE detector performs with 0% false alarm
    ### hence we take 40
    if algo =='ACE':    
        count = 0
        truth_spectra = []
        for x,y,z in truth_list:
            if count == 40:
                break
            tmp = input_img[x,y]
            tmp = np.reshape(tmp,(tmp.size))
            truth_spectra.append(tmp)
            count = count + 1
        truth_spectra = np.asarray(truth_spectra)    
        
        kmeans = KMeans(n_clusters=10, random_state=42)
        kmeans.fit(truth_spectra)
        truth_spectra = kmeans.cluster_centers_
        
        # Normalize spectra (תוצאות ירדו)
        #truth_spectra = truth_spectra / np.linalg.norm(truth_spectra, axis=1, keepdims=True)
    
    elif algo == 'SAM':
        truth_spectra = []
        for x, y, z in truth_list:
            tmp = input_img[x, y]
            tmp = np.reshape(tmp, (tmp.size))
            truth_spectra.append(tmp)
        truth_spectra = np.mean(truth_spectra, axis=0)  # Average spectrum
        
        all_spectra = np.copy(truth_spectra)
        if len(all_spectra.shape) == 1:  # If it's 1D, reshape to 2D
            all_spectra = all_spectra.reshape(-1, 1)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(all_spectra)
        all_spectra = kmeans.cluster_centers_
        
    ### SPy Matched filer class takes one target spectra as input, here passing the first one
    else:
        truth_spectra = input_img[truth_list[0][0], truth_list[0][1]]
        truth_spectra = np.reshape(truth_spectra,(truth_spectra.size))
        
        all_spectra = np.copy(truth_spectra)
        if len(all_spectra.shape) == 1:  # If it's 1D, reshape to 2D
            all_spectra = all_spectra.reshape(-1, 1)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(all_spectra)
        all_spectra = kmeans.cluster_centers_
    
    print("ground truth spectra obtained!")
    
    return truth_arr, truth_spectra
    
#-------------------SAM algorithm-----------------#

def calculate_sam(input_image, target_spectrum):

    # Normalize the target spectrum
    target_norm = np.linalg.norm(target_spectrum)

    if target_norm == 0:
        raise ValueError("Target spectrum has zero magnitude, which is invalid for SAM calculation.")

    # Get image dimensions
    rows, cols, bands = input_image.shape

    # Initialize output SAM image
    sam_image = np.zeros((rows, cols))

    # Iterate over each pixel in the image
    for i in range(rows):
        for j in range(cols):
            # Extract the pixel spectrum
            pixel_spectrum = input_image[i, j, :]
            
            # Normalize the pixel spectrum
            pixel_norm = np.linalg.norm(pixel_spectrum)

            # Avoid division by zero
            if pixel_norm == 0:
                sam_image[i, j] = np.pi / 2  # Maximum angle (90 degrees)
                continue

            # Compute the dot product between the target and pixel spectrum
            dot_product = np.dot(target_spectrum, pixel_spectrum)
            
            # Compute the cosine of the angle
            cos_theta = dot_product / (target_norm * pixel_norm)
            
            # Clip cosine value to avoid numerical errors
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            
            # Compute the spectral angle (SAM value)
            sam_image[i, j] = np.arccos(cos_theta)

    return sam_image

#------------IMPROVMENT ALGORITHMS--------------# 

#-----------------SAM WITH PCA------------------#

def calculate_sam_with_pca(input_image, target_spectrum, n_components=10):

    # Flatten the image to 2D (pixels x bands)
    rows, cols, bands = input_image.shape
    reshaped_img = input_image.reshape(-1, bands)

    # Apply PCA to reduce dimensionality
    print("Applying PCA...")
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(reshaped_img)
    print(f"PCA completed! Retained variance: {sum(pca.explained_variance_ratio_):.2f}")

    # Transform the target spectrum using PCA
    reduced_target_spectrum = pca.transform(target_spectrum.reshape(1, -1))[0]

    # Normalize the reduced target spectrum
    target_norm = np.linalg.norm(reduced_target_spectrum)
    if target_norm == 0:
        raise ValueError("Reduced target spectrum has zero magnitude, which is invalid for SAM calculation.")

    # Initialize the SAM score image
    sam_image = np.zeros((rows, cols))

    # Compute SAM scores using the reduced dimensions
    for i in range(rows):
        for j in range(cols):
            # Extract the pixel spectrum in the reduced space
            pixel_spectrum = reduced_data[i * cols + j, :]
            pixel_norm = np.linalg.norm(pixel_spectrum)

            # Avoid division by zero
            if pixel_norm == 0:
                sam_image[i, j] = np.pi / 2  # Maximum angle (90 degrees)
                continue

            # Compute the cosine similarity
            dot_product = np.dot(reduced_target_spectrum, pixel_spectrum)
            cos_theta = dot_product / (target_norm * pixel_norm)

            # Clip cosine value to avoid numerical errors
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            # Compute the spectral angle
            sam_image[i, j] = np.arccos(cos_theta)

    return sam_image




#----------------------------SDF--------------------------#

def fast_sdf(input_img, window_size=3):

    # Ensure the input is a NumPy array
    if not isinstance(input_img, np.ndarray):
        input_img = np.array(input_img.load())

    # Precompute square of the input image
    input_squared = input_img ** 2

    # Apply uniform filter (mean filter) to compute local mean and mean squared
    mean = uniform_filter(input_img, size=window_size, mode='constant')
    mean_squared = uniform_filter(input_squared, size=window_size, mode='constant')

    # Calculate standard deviation using the formula: std = sqrt(mean_squared - mean^2)
    std_dev = np.sqrt(mean_squared - mean ** 2)
    
    return std_dev

### apply the detector and get scores
### modify this function to replace SPy library detectors with your own detector algorithm(s)
def ApplyDetector(input_img, truth_spectra, algo, mask_img):
    print("Applying Detector:", algo)
    if not isinstance(input_img, np.ndarray):
        input_arr = np.copy(np.asarray(input_img.load()))
    else:
        input_arr = np.copy(input_img)
    
    mask_arr = np.copy(np.asarray(mask_img.load()))
    if np.isnan(input_arr).any():
        print("NaN values detected in the input array. Replacing with zeros.")
        input_arr = np.nan_to_num(input_arr)  # Replace NaN with zeros
    
    mask_list = GetMaskLocation(mask_arr)
    
    ## setthe masked locations to zero
    for x,y in mask_list:
        input_arr[x,y] = 0.0
    
    ### the detector func tion call to Spy class
    ### you can implement your own detector here
    if(algo == 'ACE'): 
        scores = sp.ace(input_arr, truth_spectra)
    elif algo == "SAM":
        scores = calculate_sam_with_pca(input_arr, truth_spectra)
    else :
        scores = sp.matched_filter(input_arr, truth_spectra)

    # sp.imshow(scores)
    
    print("Done running:", algo)

    return scores


##### get ground truth pixels locations: return as a list 
def GetTargetPixelList(truth_arr):
    
    print("Getting target pixel locations...")
    
    # # get the location of non zero pixels in ground truth
    target_px_in_truth = np.transpose(np.nonzero(truth_arr)).copy() 

    # # convert it into a list
    target_list = target_px_in_truth.tolist()
    
    ### fix the dimension
    for val in target_list:
        val.pop(-1)
        
    print("Done!")
    return target_list


##### get masked pixel locations: return as a list 
def GetMaskLocation(mask_arr):
    
    print("Getting mask locations...")
    
    
    # # get the location of non zero pixels in ground truth
    target_px_in_mask = np.transpose(np.where(mask_arr < 1)).copy() 

    # # convert it into a list
    target_list = target_px_in_mask.tolist()
    
    ### fix the dimension
    for val in target_list:
        val.pop(-1)
        
    print("Done!")
    return target_list


### run the steps for target detection
### this is the high level function that is called in main()
def RunTargetDetection(input_img, truth_img, plot_log, print_val, algo, mask_img, apply_sdf=False):
   
    if apply_sdf: #SDF only improves gray curve!!!
        print("Applying Standard Deviation Filter (SDF)...")
        input_img = fast_sdf(input_img)
        print("SDF applied successfully.")
    else:
        print("Skipping SDF preprocessing...")

    # Proceed with target detection using the preprocessed image
    truth_arr, truth_spectra = GetGroundTruthSpectra(input_img, truth_img, algo)  # Get target spectra
    scores = ApplyDetector(input_img, truth_spectra, algo, mask_img)  # Detector scores
    target_px_list = GetTargetPixelList(truth_arr)  # Target pixel locations
    
    ############ sort the scores and return the indices of (x,y) locations sorted by score (sorted high-low)
    
    print("Sorting scores...")
    
    score_list = []

    for index, val in np.ndenumerate(scores):
        # print(index, val)
        t= (index, val)
        score_list.append(t)
    
    if algo == "SAM":
        sorted_list = sorted(score_list, key=lambda x: x[1])  # Ascending for SAM
    else:
        sorted_list = sorted(score_list, key=lambda x: x[1], reverse=True)  # Descending for ACE/MF 

    sorted_indices = [list(x[0]) for x in sorted_list] ### extract (x,y) coordinates  as list of lists

    print("Done!")
    
    return target_px_list, sorted_indices



### this function will plot the ROC curve(s) according to selection in main()
### the function uses sorted scores (hihg-low) to plot the ROC curve(s)
### the detection scores image is sorted based on scores, then each score location is checked agianst the target pixel locations
### and PD and PFA is calculated: if current location is in taret pixels location list: +1 to target found else +1 to false alarm
##### Plot the ROC curve
def PlotROCCurve(target_list, candidate_list, plot_log, algo, t_class):
    if t_class == "gray":
        color = 'gray'
    elif t_class == "green":
        color = 'green'
        
    x_pfa = 0
    y_pd = 0
    
    PFA_list = []
    PD_list = []
    
    for location in candidate_list:
        if location in target_list: 
            y_pd = y_pd + 1  # True positive
        else:
            x_pfa = x_pfa + 1  # False positive
        
        PD_list.append(y_pd)
        PFA_list.append(x_pfa)
        
    # Normalize
    pd_max = max(PD_list)
    pfa_max = max(PFA_list)
    PFA_list2 = [float(pfa) / pfa_max for pfa in PFA_list]
    PD_list2 = [float(pd) / pd_max for pd in PD_list]
    
    # Compute and print AUC
    auc_value = auc(PFA_list2, PD_list2)
    print(f"Area Under the Curve (AUC): {auc_value:.4f}")
    
    # Plot ROC curve
    fd = 0
    plt.figure(fd)
    
    if plot_log:
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('PFA_Log')
        plt.ylabel('PD_Log')
        plt.title(f'Detector: {algo.upper()} ')
    else:
        plt.xlabel('PFA')
        plt.ylabel('PD')
        plt.title(f'Detector: {algo.upper()} with PCA')
    
    roc_plot, = plt.plot(PFA_list2, PD_list2, color)
    roc_plot.set_label(t_class)
    plt.legend()

    fd = fd + 1



def main():

    ### select ONE of the three following option to run target detection on target(s)
    ### target class: gray/green/both  
    
    t_class = "both"
    #t_class = "gray"
    #t_class = "green"

    ### turn on/off log scale plots
    plot_log = False 
    #plot_log = True
    
    ### select the detector to use: ACE / MF
    ### add your own method name here
    #detector_algo = "ACE"
    detector_algo = "MF"
    #detector_algo = "SAM"
 

    ### folder numnber: use ONE
        
    #f_num = "1428"
    f_num = "1453"
    #f_num = "1643"
    # f_num = "1702"
    
    ### this helps to debug
    print_val = False

    #### read the HSI image and the provided ground truth images here
    
    
    ### read the source HSI image here
    input_img = sp.envi.open("C:/Users/dorne/OneDrive/Desktop/Stan_Proj/1_source_data/" + f_num + "/raw_" + f_num + ".hdr") 
    
    
    ### read the ground truth imaage data for each target class (gray, green)
    
    truth_img_gray = []
    truth_img_green = []
    mask_img_gray = []
    mask_img_green = []
    
    if t_class == "both": 
        truth_img_gray = sp.envi.open("C:/Users/dorne/OneDrive/Desktop/Stan_Proj/2_detection_data/truth_images/" + f_num + "/gray_class_image.hdr") 
        truth_img_green = sp.envi.open("C:/Users/dorne/OneDrive/Desktop/Stan_Proj/2_detection_data/truth_images/" + f_num + "/green_class_image.hdr") 
        ### read the mask images for each target class
        mask_img_gray = sp.envi.open("C:/Users/dorne/OneDrive/Desktop/Stan_Proj/2_detection_data/" + f_num + '/' + detector_algo + "/gray/gray_mask.hdr") 
        mask_img_green = sp.envi.open("C:/Users/dorne/OneDrive/Desktop/Stan_Proj/2_detection_data/" + f_num + '/' + detector_algo + "/green/green_mask.hdr") 
        
    elif t_class == "gray":
        truth_img_gray = sp.envi.open("C:/Users/dorne/OneDrive/Desktop/Stan_Proj/2_detection_data/truth_images/" + f_num + "/gray_class_image.hdr") 
        mask_img_gray = sp.envi.open("C:/Users/dorne/OneDrive/Desktop/Stan_Proj/2_detection_data/" + f_num + '/' + detector_algo + "/gray/gray_mask.hdr")
    elif t_class == "green":
        truth_img_green = sp.envi.open("C:/Users/dorne/OneDrive/Desktop/Stan_Proj/2_detection_data/truth_images/" + f_num + "/green_class_image.hdr")  
        mask_img_green = sp.envi.open("C:/Users/dorne/OneDrive/Desktop/Stan_Proj/2_detection_data/" + f_num + '/' + detector_algo + "/green/green_mask.hdr")
        
    

    
    # sp.imshow(mask_img)
    
    ### return values from target detection function
    gray_targets = []
    gray_indices = []
    green_targets  = []
    green_indices = []
    
    
    ### runs ACE/MF on selected target(s) and plots the ROC curve(s)
    ## to test your own target detection algorithm(s) modify the functions mentioned at the beginning of this soruce code file
    
    if t_class == "gray":
        gray_targets, gray_indices = RunTargetDetection(input_img, truth_img_gray, plot_log, print_val, detector_algo, mask_img_gray, apply_sdf=True)
        PlotROCCurve(gray_targets, gray_indices, plot_log, detector_algo, 'gray')
    elif t_class == "green":
        green_targets, green_indices = RunTargetDetection(input_img, truth_img_green, plot_log, print_val, detector_algo, mask_img_green, apply_sdf=False)
        PlotROCCurve(green_targets, green_indices, plot_log, detector_algo, 'green')
    else:  # Run for both classes
        gray_targets, gray_indices = RunTargetDetection(input_img, truth_img_gray, plot_log, print_val, detector_algo, mask_img_gray, apply_sdf=True)
        green_targets, green_indices = RunTargetDetection(input_img, truth_img_green, plot_log, print_val, detector_algo, mask_img_green, apply_sdf=False)
        PlotROCCurve(gray_targets, gray_indices, plot_log, detector_algo, 'gray')
        PlotROCCurve(green_targets, green_indices, plot_log, detector_algo, 'green')


    ### plot the ROC curve(s)
    plt.show()
   
    
main()

###############################################################################