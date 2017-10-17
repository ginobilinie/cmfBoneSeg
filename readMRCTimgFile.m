%the infant brain images, the are three type images: T1( xx_cbq.hdr), T2
%(xx_cbq-T2.hdr), label(xx-ls-corrected.hdr)
%I convert the label images to 0,1,2,3
function readMRCTimgFile()
%addpath('/home/dongnie/Desktop/Caffes/software/NIfTI_20140122');
%addpath('/home/dongnie/Desktop/Caffes/software/REST_V1.9_140508/REST_V1.9_140508');
% instensity image
d1=5;
d2=184;
d3=152;
len=13;
step=1;
rate=1;
path='./';
ids=[2 3 4 5 8 9 10 13];
threshold=1600;
d=[d1,d2,d3];
%flipIDs=[12 13 15 16 18 20 21 22];
path='/home/dongnie/warehouse/BrainCTData/';
dirname='P1/';
flipdirname='P1_Flip/';
mrfilename='mr1_processed_resampled_p2_cropped_processed_histmatchedp2.hdr';
ctfilename='ct1_processed_resampled_p2_cropped_processed.hdr';

flipmrfilename='mr1_processed_resampled_p2_cropped_processed_histmatchedp2_flip.hdr';
flipctfilename='ct1_processed_resampled_p2_cropped_processed_flip.hdr';
%labelfilename='mask.hdr';
%ctfilename='a.hdr';
for i=1:length(ids)
    id=ids(i);
    
%% for original images
    currdirname=strrep(dirname,'1',sprintf('%d',id));
%     ctfilename=sprintf('prostate_%dto1_CT.nii',id);
%     [ctimg cthead]=rest_ReadNiftiImage([path,ctfilename]);
    info = analyze75info([path,currdirname,mrfilename]);
    mrimg =single(analyze75read(info));%
    %normalize it,learned from Andrew's course, haha
    mu=mean(mrimg(:));
    maxV=max(mrimg(:));
    minV=min(mrimg(:));
    mrimg=(mrimg-mu)/(maxV-minV);
    
    info = analyze75info([path,currdirname,ctfilename]);
    ctimg = analyze75read(info);%
    
%    [mrMat mrHead]=rest_ReadNiftiImage('coT1MPRAGETRAiso10.nii');

    
    temp=ctimg;
    temp(find(ctimg>1600))=1;
    temp(find(ctimg<=1600))=0;
    
    labelimg=temp;
    oid=sprintf('%d',id);
    [matOut,cnt]=cropCubic(mrimg,ctimg,labelimg,oid,d,step,rate);
    preMat=matOut;
     save(sprintf('preSub_%d.mat',id),'preMat','labelimg');
%% for flip images    
      flipcurrdirname=strrep(flipdirname,'1',sprintf('%d',id));
%     ctfilename=sprintf('prostate_%dto1_CT.nii',id);
%     [ctimg cthead]=rest_ReadNiftiImage([path,ctfilename]);
    info = analyze75info([path,flipcurrdirname,flipmrfilename]);
    flipmrimg = single(analyze75read(info));%
    %normalize it,learned from Andrew's course, haha
    mu=mean(flipmrimg(:));
    maxV=max(flipmrimg(:));
    minV=min(flipmrimg(:));
    flipmrimg=(flipmrimg-mu)/(maxV-minV);
    
    
    info = analyze75info([path,flipcurrdirname,flipctfilename]);
    flipctimg = analyze75read(info);%
    temp=flipctimg;
    temp(find(flipctimg>1500))=1;
    temp(find(flipctimg<=1500))=0;
    
    fliplabelimg=temp;
    flipid=sprintf('flip%d',id);
    [matOut,cnt]=cropCubic(flipmrimg,flipctimg,fliplabelimg,flipid,d,step,rate);
    flippreMat=matOut;
    DRs(i)=computeDR(matOut,fliplabelimg)
      save(sprintf('preSub_%d_flip.mat',id),'flippreMat','fliplabelimg');

%     info = analyze75info([currdirname,labelfilename]);
%     labelimg = analyze75read(info);%
%     
%     info = analyze75info([currdirname,labelfilename]);
%     labelimg = analyze75read(info);%
    
%     labelimg(find(labelimg>200))=3;%white matter
%     labelimg(find(labelimg>100))=2;%gray matter
%     labelimg(find(labelimg>4))=1;%csf
    
%      mrfilename=sprintf('prostate_%dto1_MRI.nii',id);
%     [mrimg mrhead]=rest_ReadNiftiImage([path,mrfilename]);
%     
%     words=regexp(t1filename,'_','split');
%     word=words{1};
%     word=lower(word);
%     saveFilename=sprintf('%s',word);
    %crop areas
    
%     cnt=cropCubic(mrimg,ctimg,labelimg,id,d,step,rate);
    
    
  
    
end
% 
% t1filename='NORMAL01_cbq.hdr';
% info = analyze75info([path,t1filename]);
% t1img = analyze75read(info);
% 
% t2filename='NORMAL01_cbq-T2.hdr';
% info = analyze75info([path,t2filename]);
% t2img = analyze75read(info);
% 
% labelfilename='NORMAL01-ls-corrected.hdr';
% info = analyze75info([path,labelfilename]);
% labelimg = analyze75read(info);

return


%crop width*height*length from mat,and stored as image
%note,matData is 3 channels, matSet is 1 channel
%d: the patch size
function [matOut,cubicCnt]=cropCubic(matFA,matCT,matSeg,fileID,d,step,rate)   
    eps=1e-2;
    if nargin<6
    	rate=1/4;
    end
    if nargin<5
        step=4;
    end
    if nargin<4
        d=16;
    end
    [row,col,len]=size(matFA);
%% make the size to be better by padding zeros
    tmat=zeros(row,184,152);
    tmat(:,1:181,1:149)=matFA;
    matFA=tmat;
    
    tmat=zeros(row,184,152);
    tmat(:,1:181,1:149)=matSeg;
    matSeg=tmat;
      [row,col,len]=size(matFA);
    %[rowData,colData,lenData]=size(matT1);
   
    %if row~=rowData||col~=colData||len~=lenData
     %   fprintf('size of matData and matSeg is not consistent\n');
     %   exit
    %end
    cubicCnt=0;
%     fid=fopen('trainSkullStripping_list.txt','a');

    for i=1:step:row-d(1)+1
        for j=1:step:col-d(2)+1
            for k=1:step:len-d(3)+1%there is no overlapping in the 3rd dim
                volSeg=single(matSeg(i:i+d(1)-1,j:j+d(2)-1,k:k+d(3)-1));
                cubicCnt=cubicCnt+1;
                if sum(volSeg(:))<eps%all zero submat
                    matOut(cubicCnt,:,:)=volSeg;
                    continue;
                end
                volFA=single(matFA(i:i+d(1)-1,j:j+d(2)-1,k:k+d(3)-1));
                %% for 3D data
%                 trainFA(:,:,:,1,cubicCnt)=volFA;
%                 trainSeg(:,:,:,1,cubicCnt)=volSeg;       
                %% for 2d dataset
                tempmatMR=squeeze(volFA);
                temppremat=test_3dBrainFCN(tempmatMR);
                matOut(cubicCnt,:,:)=temppremat;
                %trainFA(:,:,1,cubicCnt)=squeeze(volFA);
                %trainSeg(:,:,1,cubicCnt)=squeeze(volSeg);    
            end
        end
    end
%      trainFA=single(trainFA);
%      trainSeg=int8(trainSeg);
     matOut=int8(matOut);
%      h5create(sprintf('train_%s.hdf5',fileID),'/dataMR',size(trainFA),'Datatype','single');
%      h5write(sprintf('train_%s.hdf5',fileID),'/dataMR',trainFA);
%      h5create(sprintf('train_%s.hdf5',fileID),'/dataSeg',size(trainSeg),'Datatype','int8');
%      h5write(sprintf('train_%s.hdf5',fileID),'/dataSeg',trainSeg);
%      clear trainFA;
%      clear trainSeg;
%      fprintf(fid,sprintf('train_%s.hdf5\n',fileID));	
%      fclose(fid);
return
