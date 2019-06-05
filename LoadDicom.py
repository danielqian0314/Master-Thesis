%matplotlib inline
#%% Required Library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom as dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#%% Get ids of patients
INPUT_FOLDER = './data/'
#INPUT_FOLDER = "./med_data/Melanom/"
patients = os.listdir(INPUT_FOLDER)
patients.sort()

#%% get path of dicom files in a given dir
def findDicomfile(path):
    lstFilesDCM = []
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            lstFilesDCM.append(os.path.join(dirName,filename))
    return lstFilesDCM
#%% load all dicom slices of a given patient and image type
def load_scan(patient_id, image_type):
    lstFilesDCM=findDicomfile(INPUT_FOLDER + patient_id+"/Texturanalyse/"+patient_id+" "+ image_type)
    slices = [dicom.read_file(s) for s in lstFilesDCM]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    print("number of slices:", len(slices))

    return slices
#%% load the compressed dicom mask of a given patient and image type
def load_scan_mask(patient_id, image_type):
    paths = findDicomfile(INPUT_FOLDER + patient_id+"/Texturanalyse/"+patient_id+" "+ image_type+ " Läsion")
    print(paths)
    image =dicom.read_file(paths[0])

    return image


#%% get a 3d pixel array from all slices
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    image[image == -2000] = 0

    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    print("size of pixel array:", image.shape)

    return np.array(image, dtype=np.int16)


#%% show certain slice
def showSlice(patient, patient_pixels, slice):
    pat_name = patient[slice].PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name...:", display_name)
    print("Patient id.......:", patient[slice].PatientID)
    print("Modality.........:", patient[slice].Modality)
    print("Study Date.......:", patient[slice].StudyDate)
    if 'PixelData' in patient[slice]:
        rows = int(patient[slice].Rows)
        cols = int(patient[slice].Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
        rows=rows, cols=cols, size=len(patient[slice].PixelData)))
    if 'PixelSpacing' in patient[slice]:
        print("Pixel spacing....:", patient[slice].PixelSpacing)
    print("Slice location...:", patient[slice].get('SliceLocation', "(missing)"))
    plt.imshow(patient_pixels[slice], cmap=plt.cm.gray)
    plt.show()

#%% show mask
def showMaskInfo(patient_mask):
    pat_name = first_patient_mask.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name...:", display_name)
    print("Patient id.......:", first_patient_mask.PatientID)
    print("Modality.........:", first_patient_mask.Modality)
    print("Study Date.......:", first_patient_mask.StudyDate)
    print("Mask size........:", first_patient_mask.pixel_array.shape)

#%% 
first_patient_CT_mask = load_scan_mask(patients[0], "CT")
showMaskInfo(first_patient_CT_mask)
#%% 
first_patient_CT = load_scan(patients[0], "CT")
first_patient_CT_pixel = get_pixels_hu(first_patient_CT)
#%%
showSlice(first_patient_CT,first_patient_CT_pixel,10)















#%%
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
print("Shape before resampling\t", first_patient_pixels.shape)
print("Shape after resampling\t", pix_resampled.shape)

#%%
def plot_3d(image, threshold=-300):

    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)

    verts, faces = measure.marching_cubes_classic(p,threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


#%%
lassion_path=findDicomfile(INPUT_FOLDER + "001/Texturanalyse/001 CT Läsion")[0]
#%%
first_patient_Lassion = dicom.read_file(lassion_path)
first_patient_Lassion.pixel_array.shape
#%%
from mayavi import mlab
import vtk
first_patient_Lassion = dicom.read_file(lassion_path)
mlab.volume_slice(first_patient_Lassion.pixel_array)
mlab.show()

