import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  
import scipy.io as scio
from scipy import ndimage as nd

path='./'

#this function is used to compute the dice ratio
def dice(im1, im2,tid):
    im1=im1==tid #make it boolean
    im2=im2==tid #make it boolean
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc=2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc

def main():
    id=2
    mrPath=os.path.join(path,'P%d/mr1_processed_resampled_p2_cropped_processed_histmatchedp2.hdr'%id)
    strippedPath=os.path.join(path,'P%d/bet_sub2.nii.gz'%id)
    maskPath=os.path.join(path,'P%d/mask.hdr'%id)
    gtOrg=sitk.ReadImage(mrPath)
    mrMat=sitk.GetArrayFromImage(gtOrg)
    
    gtOrg=sitk.ReadImage(strippedPath)
    strippedMat=sitk.GetArrayFromImage(gtOrg)
    
    gtOrg=sitk.ReadImage(maskPath)
    gtMat=sitk.GetArrayFromImage(gtOrg)
    
    print mrMat.shape,strippedMat.shape
    
    boneMat=mrMat-strippedMat
    boneMat[boneMat>0.5]=1
    
    dsc=dice(gtMat,boneMat,1)
    print 'dice for id',id,' is ',dsc
    #gtMat=np.transpose(gtMat,(2,1,0))
    gtVol=sitk.GetImageFromArray(boneMat)
    sitk.WriteImage(gtVol,'P%d/bone_sub2.nii.gz'%id)

    

if __name__ == '__main__':     
    main()