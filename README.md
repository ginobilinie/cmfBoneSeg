# cmfBoneSeg
Segmentation of Craniomaxillofacial Bony Structures from MRI with A Cascade Deep Learning Framework. There are an amount of plastic surgery every year. Traditionally, medical doctors use CT as the modality to model the CMF bone structure. While CT is radioactive, so we plan to use MRI which is a safe and non-invasive protocol as the modality to assess the CMF bone model. We propose to use a cascade DL framework, which can adaptively pay more attention to the difficulty regions (mainly boundaries), to assess the CMF bone model.

We attach all the codes in this repository, if you think it is helpful for you, please cite:

"Nie, D., Wang, L., Trullo, R., Li, J., Yuan, P., Xia, J., & Shen, D. (2017, September). Segmentation of Craniomaxillofacial Bony Structures from MRI with a 3D Deep-Learning Based Cascade Framework. In International Workshop on Machine Learning in Medical Imaging (pp. 
266-273). Springer, Cham."

If you are interested in our work but have any problems, please contact me via ginobilinie at gmail.com
1. extractSsPatch4SegMedImg.py: extracting patches for 1st 3d Unet
2. extractPatches4CNN2.py: extracting pactches for the 2nd context-cnn with uncertainty sampling
3. evalCaffeCMFSeg4MedImg.py: evaluation for the 1st 3d U-Net
4. evalCaffeCMFSegCascade4MedImg.py: evaluation for the 2nd context-cnn
5. infant_train_test_3d.prototxt or boneSeg_train_test_3d: the 3d Unet network definition file
6. infant_solver_3d.prototxt or  solver_mri_3d.txt: the solver file for training the 3d Unet network
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
