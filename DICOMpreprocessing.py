
# coding: utf-8

import numpy as np
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from collections import OrderedDict

SCRIPT_DIR = os.getcwd()
INPUT_FOLDER = './Input/Sample_Images/'
OUTPUT_FOLDER = './Input/Sample_training_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()


# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -
                                 slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation -
                                 slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


# In[4]:

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


# In[5]:

def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing


# pat1 = load_scan(INPUT_FOLDER + patients[0])
# pat1_pixels = get_pixels_hu(pat1)
# plt.hist(pat1_pixels.flatten(), bins=80, color='c')
# plt.xlabel("Hounsfield Units (HU)")
# plt.ylabel("Frequency")
# plt.show()
#
# # Show some slice in the middle
# plt.imshow(pat1_pixels[80], cmap=plt.cm.gray)
# plt.show()

#
# pix_resampled, spacing = resample(pat1_pixels, pat1, [1, 1, 1])
# print("Shape before resampling\t", pat1_pixels.shape)
# print("Shape after resampling\t", pix_resampled.shape)


patient_dict = OrderedDict()

for i in patients:
    patient_dict[i] = load_scan(INPUT_FOLDER + i)


for key in patient_dict:
    if not os.path.exists(OUTPUT_FOLDER + str(key)):
        os.makedirs(OUTPUT_FOLDER + str(key))
    print('Successfully created a folder for ' + str(key) + '!')
    os.chdir(OUTPUT_FOLDER + str(key))
    pat_pixels = get_pixels_hu(patient_dict[key])
    print('Finished converting pixels to hu for ' + str(key) + '!')
    pix_resampled, spacing = resample(pat_pixels, patient_dict[key], [1, 1, 1])
    print('Finished resampling the pixels for ' + str(key) + '!')
    for x in range(0, len(pix_resampled)):
        plt.imshow(pix_resampled[x], cmap=plt.cm.gray)
        plt.savefig('slice' + str(x) + '.png')
        print('Saving slice' + str(x) + '...')
        plt.close()
    print('Done processing ' + str(key) + '!')
    os.chdir(SCRIPT_DIR)

# call this during training

# MIN_BOUND = -1000.0
# MAX_BOUND = 400.0
#
#
# def normalize(image):
#     image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
#     image[image > 1] = 1.
#     image[image < 0] = 0.
#     return image
#
#
# PIXEL_MEAN = 0.25
#
#
# def zero_center(image):
#     image = image - PIXEL_MEAN
#     return image
