#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

We extract patches for the 2nd CNN based on the result of the 1st CNN.
07/31/2017
@author: Dong Nie
"""

import numpy as np
import scipy.io as sio
import os 
import h5py    
import SimpleITK as sitk
import random as rd
#import augment as aug
import glob

dFA=[15,15,15]
dSeg=[1,1,1]

subsetID='00'
str1='/shenlab/lab_stor3/dongnie/tianchi_data/Mat/train_subset0*'
subject=glob.glob(str1)
thresholdProb=0.7

def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = np.float32((image - MIN_BOUND)) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    #newimg = (newimg*255).astype('uint8')
    return image
all_num =0

'''
Actually, we donot need it any more,  this is useful to generate hdf5 database
'''
def extractPatches4OneSubject(matFA, matSeg, matPred, matProb, fileID, d, step, rate):
  
    rate1 = 1.0/2
    rate2 = 1.0/4
    [row,col,leng] = matFA.shape
    cubicCnt = 0
    estNum = 20000
    trainFA = np.zeros([estNum,1, dFA[0],dFA[1],dFA[2]],np.float16)
    trainSeg = np.zeros([estNum,1,dSeg[0],dSeg[1],dSeg[2]],dtype=np.int8)
    trainProb = np.zeros([estNum,1, dFA[0],dFA[1],dFA[2]],np.float16)

#     trainFA2D=np.zeros([estNum, dFA[0],dFA[1],dFA[2]],np.float16)
#     trainSeg2D=np.zeros([estNum,dSeg[0],dSeg[1],dSeg[2]],dtype=np.int8)
    print 'trainFA shape, ',trainFA.shape
    #to padding for input
    margin1 = (dFA[0]-dSeg[0])/2
    margin2 = (dFA[1]-dSeg[1])/2
    margin3 = (dFA[2]-dSeg[2])/2
    cubicCnt = 0
    marginD = [margin1,margin2,margin3]
    print 'matFA shape is ',matFA.shape
    matFAOut = np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    print 'matFAOut shape is ',matFAOut.shape
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA
    matSegOut = np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matSeg
    matPredOut = np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    matPredOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matPred
    matProbOut = np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    matProbOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matProb
    
    # for mageFA, enlarge it by padding
    if margin1!=0:
        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]
    # for matseg, enlarge it by padding
    if margin1!=0:
        matSegOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matSegOut[row+marginD[0]:matSegOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg[matSeg.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matSegOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matSeg[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matSegOut[marginD[0]:row+marginD[0],col+marginD[1]:matSegOut.shape[1],marginD[2]:leng+marginD[2]]=matSeg[:,matSeg.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matSeg[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matSegOut.shape[2]]=matSeg[:,:,matSeg.shape[2]-1:leng-marginD[2]-1:-1]
    # for the prediction map     
    if margin1!=0:
        matPredOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matPred[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matPredOut[row+marginD[0]:matPredOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matPred[matPred.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matPredOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matPred[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matPredOut[marginD[0]:row+marginD[0],col+marginD[1]:matPredOut.shape[1],marginD[2]:leng+marginD[2]]=matPred[:,matPred.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matPredOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matPred[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matPredOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matPredOut.shape[2]]=matPred[:,:,matPred.shape[2]-1:leng-marginD[2]-1:-1]    
    # for the probability map     
    if margin1!=0:
        matProbOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matProb[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matProbOut[row+marginD[0]:matProbOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matProb[matProb.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matProbOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matProb[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matProbOut[marginD[0]:row+marginD[0],col+marginD[1]:matProbOut.shape[1],marginD[2]:leng+marginD[2]]=matProb[:,matProb.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matProbOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matProb[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matProbOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matProbOut.shape[2]]=matProb[:,:,matProb.shape[2]-1:leng-marginD[2]-1:-1]            
    dsfactor = rate
    #actually, we can specify a bounding box along the 2nd and 3rd dimension, so we can make it easier 
    for i in range(0,row-dSeg[0],step[0]):
        for j in range(0,col-dSeg[1],step[1]):
            for k in range(0,leng-dSeg[2],step[2]):
                volSeg = matSegOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                if np.sum(volSeg)<eps:
                    continue
                if matProbOut[i,j,k]>0.7 or matProbOut[i,j,k]<0.3 and matPredOut[i,j,k] == matSegOut[i,j,k]: # we only consider those uncertain regions or those uncorrected classifed regions
                    continue
                cubicCnt = cubicCnt+1
                #index at scale 1
            
                
                volFA = matFAOut[i:i+dFA[0],j:j+dFA[1],k:k+dFA[2]]
                volProb = matProbOut[i:i+dFA[0],j:j+dFA[1],k:k+dFA[2]]
              
                trainFA[cubicCnt,0,:,:,:] = volFA #32*32*32
                trainSeg[cubicCnt,0,:,:,:] = volSeg#24*24*24
                trainProb[cubicCnt,0,:,:,:] = volProb#24*24*24

#                 trainFA2D[cubicCnt,:,:,:]=volFA #32*32*32
#                 trainSeg2D[cubicCnt,:,:,:]=volSeg#24*24*24


    trainFA = trainFA[0:cubicCnt,:,:,:,:]
    trainSeg = trainSeg[0:cubicCnt,:,:,:,:]
    trainProb = trainProb[0:cubicCnt,:,:,:,:]

#     trainFA2D=trainFA2D[0:cubicCnt,:,:,:]
#     trainSeg2D=trainSeg2D[0:cubicCnt,:,:,:]

    with h5py.File('./traincnn2_%s.h5'%fileID,'w') as f:
        f['dataMR'] = trainFA
        f['dataSeg'] = trainSeg
        f['dataProb'] = trainProb
#         f['dataMR2D']=trainFA2D
#         f['dataSeg2D']=trainSeg2D
     
    with open('./traincnn2_cmfBoneSeg_list.txt','a') as f:
        f.write('/shenlab/lab_stor3/dongnie/cmfBoneSeg/cmfH5/traincnn2_%s.h5\n'%fileID)
    return cubicCnt


def extractPatches():
    estNum=100000
    patchID=0
    trainFA=np.zeros([estNum,1, dFA[0],dFA[1],dFA[2]],np.float16)
    trainSeg=np.zeros([estNum,1,dSeg[0],dSeg[1],dSeg[2]],dtype=np.int8)
    for sub_dir in subject:
        str2=sub_dir+'/'
        subject_indiv=os.listdir(str2)
        for i in range(len(subject_indiv)):
            name_sub=subject_indiv[i]
            spli=name_sub.split(".") 
            name_now=spli[0]
            if spli[1]!='mat':
                continue
            full_name=str2+name_sub
            matinfo = sio.loadmat(full_name)
            image=matinfo['lungData']
            print 'image max is, ',np.max(image)

            mask=matinfo['noduleMask']
            ini=matinfo['Region']
            
            #full_name='/shenlab/lab_stor3/dongnie/tianchi_code/fp/2_candidate/cand_deepmedic_score0.1_size4/'+name_now+'.h5'
            full_name='/shenlab/lab_stor3/dongnie/tianchi_code/fp/2_candidate/cnn1/'+name_now+'.h5'
            print full_name

            gt = h5py.File(full_name,'r') 
            probs=gt['prob'][:] #we store the predicted probability in the h5 as well
            preLabels=gt['preLabel'][:] #we store the predicted label in the h5 as well
            labels=gt['labels'][:]
            coords=gt['cand_coords_1mm'][:]
            coords=coords-ini[0,0::2]
            coords=np.array(coords,dtype='int')
    #        labels=(labels>0).astype(np.int)
            gt.close()
            #print('sub_dir ' + str(sub_dir)+' / ' +str(len(subject))+', sub_im '+str(i)+ ', traiing num is '+str(len(coords)))
 
    
    #        assert(0)
            [ww,hh,dd]=image.shape
    #        
            for k in range(len(coords)):
                    #pass the high confidence of negative values
                    if preLabels[k]==0 and labels[k]==0 and probs[k]>thresholdProb:
                        continue
                    patch_48=np.zeros((48,48,48))
                    print 'the coords are', coords[k][0],coords[k][1],coords[k][2]
                    if coords[k][0]-24<0 or coords[k][0]+24>ww or coords[k][1]-24<0 or coords[k][1]+24>hh or coords[k][2]-24<0 or coords[k][2]+24>dd:
                        continue
                    patch1=image[coords[k][0]-24:coords[k][0]+24,coords[k][1]-24:coords[k][1]+24,coords[k][2]-24:coords[k][2]+24]
                    imx=np.size(patch1,0)
                    imy=np.size(patch1,1)
                    imz=np.size(patch1,2)
                    
                    
                    patch_48[:imx,:imy,:imz]=patch1  
                    print patch1.shape
                    label=[coords[k],labels[k]]
                    trainFA[patchID,0,:,:,:]=patch1;
                    trainSeg[patchID,0,:,:,:]=labels[k]
                    patchID=patchID+1
                    
                    if labels[k]==1:
                        patch2=image[coords[k][0]-3-24:coords[k][0]-3+24,coords[k][1]-24:coords[k][1]+24,coords[k][2]-24:coords[k][2]+24]
                        if coords[k][0]-3-24<0 or coords[k][0]-3+24>ww or coords[k][1]-24<0 or coords[k][1]+24>hh or coords[k][2]-24<0 or coords[k][2]+24>dd:
                            continue
                        patch3=image[coords[k][0]+3-24:coords[k][0]+3+24,coords[k][1]-24:coords[k][1]+24,coords[k][2]-24:coords[k][2]+24]
                        if coords[k][0]+3-24<0 or coords[k][0]+3+24>ww or coords[k][1]-24<0 or coords[k][1]+24>hh or coords[k][2]-24<0 or coords[k][2]+24>dd:
                            continue
                        patch4=image[coords[k][0]-24:coords[k][0]+24,coords[k][1]-3-24:coords[k][1]-3+24,coords[k][2]-24:coords[k][2]+24]
                        if coords[k][0]-24<0 or coords[k][0]+24>ww or coords[k][1]-3-24<0 or coords[k][1]-3+24>hh or coords[k][2]-24<0 or coords[k][2]+24>dd:
                            continue
                        patch5=image[coords[k][0]-24:coords[k][0]+24,coords[k][1]+3-24:coords[k][1]+3+24,coords[k][2]-24:coords[k][2]+24]
                        if coords[k][0]-24<0 or coords[k][0]+24>ww or coords[k][1]+3-24<0 or coords[k][1]+3+24>hh or coords[k][2]-24<0 or coords[k][2]+24>dd:
                            continue
                        patch6=image[coords[k][0]-24:coords[k][0]+24,coords[k][1]-24:coords[k][1]+24,coords[k][2]-3-24:coords[k][2]-3+24]
                        if coords[k][0]-24<0 or coords[k][0]+24>ww or coords[k][1]-24<0 or coords[k][1]+24>hh or coords[k][2]-3-24<0 or coords[k][2]-3+24>dd:
                            continue
                        patch7=image[coords[k][0]-24:coords[k][0]+24,coords[k][1]-24:coords[k][1]+24,coords[k][2]+3-24:coords[k][2]+3+24]
                        if coords[k][0]-24<0 or coords[k][0]+24>ww or coords[k][1]-24<0 or coords[k][1]+24>hh or coords[k][2]+3-24<0 or coords[k][2]+3+24>dd:
                            continue
                        trainFA[patchID,0,:,:,:]=patch2;
                        trainSeg[patchID,0,:,:,:]=labels[k]  
                        patchID=patchID+1
                        trainFA[patchID,0,:,:,:]=patch3;
                        trainSeg[patchID,0,:,:,:]=labels[k]  
                        patchID=patchID+1
                        trainFA[patchID,0,:,:,:]=patch4;
                        trainSeg[patchID,0,:,:,:]=labels[k]  
                        patchID=patchID+1
                        trainFA[patchID,0,:,:,:]=patch5;
                        trainSeg[patchID,0,:,:,:]=labels[k]  
                        patchID=patchID+1
                        trainFA[patchID,0,:,:,:]=patch6;
                        trainSeg[patchID,0,:,:,:]=labels[k]  
                        patchID=patchID+1 
                        trainFA[patchID,0,:,:,:]=patch7;
                        trainSeg[patchID,0,:,:,:]=labels[k]  
                        patchID=patchID+1         
#                     patch_mask=np.zeros((48,48,48))
#                     patch2=mask[coords[k][0]-24:coords[k][0]+24,coords[k][1]-24:coords[k][1]+24,coords[k][2]-24:coords[k][2]+24]
#                     imx=np.size(patch2,0)
#                     imy=np.size(patch2,1)
#                     imz=np.size(patch2,2)
#                     patch_mask[:imx,:imy,:imz]=patch2   
#                     
#                     str3='./val_deepmedic/'+name_now+'_'+str(k)+'_crop.npy'
#                     str4='./val_deepmedic/'+name_now+'_'+str(k)+'_label.npy'
#                     str5='./val_deepmedic/'+name_now+'_'+str(k)+'_mask.npy'  
    #                np.save(str3,patch_48)
    #                np.save(str4,label)
    #                np.save(str5,patch_mask) 
     
    trainFA=trainFA[0:patchID,:,:,:,:]
    trainSeg=trainSeg[0:patchID,:,:,:,:]

    
    with h5py.File('./trainLung_%s.h5'%subsetID,'w') as f:
        f['dataMR']=trainFA
        f['dataSeg']=trainSeg
    
    with open('./trainLung_list.txt','a') as f:
        f.write('/shenlab/lab_stor3/dongnie/tianchi_code/fp/fpH5/trainLung_%s.h5\n'%subsetID)              
    #print('all traiing num is '+str(all_num))
    return patchID


def main():
    path='/shenlab/lab_stor3/dongnie/cmfBoneSeg/normals/'
    saveto='/shenlab/lab_stor3/dongnie/cmfBoneSeg/cmfH5/'
#     path='/shenlab/lab_stor3/dongnie/pelvicSeg/mrs_data/'
#     saveto='/shenlab/lab_stor3/dongnie/pelvicSeg/mrs_data/'
 
    step = [2,2,2]
    ids=[1,2,3,4,5,6,7,8,9,10,11]
    for ind in ids:
        datafilename='mri%d_resampled.nii.gz'%ind #provide a sample name of your filename of data here
        datafn=os.path.join(path,datafilename)
        labelfilename='mask%d_resampled.nii.gz'%ind  # provide a sample name of your filename of ground truth here
        labelfn=os.path.join(path,labelfilename)
        probfilename='prob%d_resampled.nii.gz'%ind  # provide a sample name of your filename of ground truth here
        probfn=os.path.join(path,probfilename)
        predfilename='pred%d_resampled.nii.gz'%ind  # provide a sample name of your filename of ground truth here
        predfn=os.path.join(path,predfilename)
        imgOrg=sitk.ReadImage(datafn)
        mrimg=sitk.GetArrayFromImage(imgOrg)
        tmpMR=mrimg
           #mrimg=mrimg
        
        labelOrg=sitk.ReadImage(labelfn)
        labelimg=sitk.GetArrayFromImage(labelOrg)
        probOrg=sitk.ReadImage(probfn)
        probimg=sitk.GetArrayFromImage(probOrg)
        predOrg=sitk.ReadImage(predfn)
        predimg=sitk.GetArrayFromImage(predOrg)
        
        rate=1
        print 'unique labels are ',np.unique(labelimg)
        print 'it comes to sub',ind
        print 'shape of mrimg, ',mrimg.shape
        
        mu=np.mean(mrimg)
        maxV, minV=np.percentile(mrimg, [99 ,1])
        std=np.std(mrimg)
        #print 'maxV,',maxV,' minV, ',minV
        #mrimg=(mrimg-mu)/(maxV-minV)
        mrimg=(mrimg-mu)/std
        
        print 'maxV,',np.ndarray.max(mrimg),' minV, ',np.ndarray.min(mrimg)

        fileID='%d'%ind
        cubicCnt=extractPatches4OneSubject(mrimg,labelimg,probimg, fileID,dFA,step,rate)
        print '# of patches is ', cubicCnt
        
        tmpMat=mrimg
        tmpLabel=labelimg
        #reverse along the 1st dimension 
        mrimg=mrimg[tmpMat.shape[0]-1::-1,:,:]
        labelimg=labelimg[tmpLabel.shape[0]-1::-1,:,:]
        fileID='%d_flip1'%ind
        cubicCnt=extractPatches4OneSubject(mrimg,labelimg, probimg, fileID,dFA,step,rate)
        #reverse along the 2nd dimension 
        mrimg=mrimg[:,tmpMat.shape[1]-1::-1,:]
        labelimg=labelimg[:,tmpLabel.shape[1]-1::-1,:]
        fileID='%d_flip2'%ind
        cubicCnt=extractPatches4OneSubject(mrimg,labelimg, probimg, fileID,dFA,step,rate)
        #reverse along the 2nd dimension 
        mrimg=mrimg[:,:,tmpMat.shape[2]-1::-1]
        labelimg=labelimg[:,:,tmpLabel.shape[2]-1::-1]
        fileID='%d_flip3'%ind
        cubicCnt=extractPatches4OneSubject(mrimg,labelimg, predimg, probimg, fileID, dFA,step,rate)

    
if __name__ == '__main__':     
    main()                
            
        
