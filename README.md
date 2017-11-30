# cmfBoneSeg
Segmentation of Craniomaxillofacial Bony Structures from MRI with A Cascade Deep Learning Framework

1. extractSsPatch4SegMedImg.py: extracting patches for 1st 3d Unet
2. extractPatches4CNN2.py: extracting pactches for the 2nd context-cnn with uncertainty sampling
3. evalCaffeCMFSeg4MedImg.py: evaluation for the 1st 3d U-Net
4. evalCaffeCMFSegCascade4MedImg.py: evaluation for the 2nd context-cnn
5. infant_train_test_3d.prototxt: the 3d Unet network definition file
6. infant_solver_3d.prototxt: the solver file for training the 3d Unet network
7. context_cnn_train_test.prototxt: the 3d context-cnn network definitiion file


For the <B>1st</B> stage: you can use the following settings, I have high confidence that you can achieve even better performance than what I wrote in my paper:

<B>Training elements:</B>
  a). train_boneSeg.sh
  b). boneSeg_train_test_3d.prototxt
  c). solver_mri_3d.prototxt

<B>Testing elements:</B>
  a). evalCaffeCMFSeg4MedImg.py
  b). boneSeg_deploy_3d1.prototxt


For the <B>2nd</B> stage: you can use the following settings, I also have high confidence you can achieve at least the same performance mentioned in my paper:
