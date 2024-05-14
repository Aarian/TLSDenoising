img = imread('boat.png');
%img = imread('lena.jpg')
size(img);
gaussian_noise_sigma = .2 ;
n=length(img);
Noise = GWN2(n,gaussian_noise_sigma);
%img_double=double(img);
img_double = im2double(img);
%img_double = img
img_noisy = log(img_double+1)+Noise;
img_double_log = log(img_double+1);

img_double = img_double_log ;


% exp_img = int8(exp((img_noisy)));
% %imshow([int8(exp_img) ;img_double;img]  ) ;
% %imshow()
% %X = int8(img) - int8(exp_img)
% figure
% imshow(int8(exp_img));
% 
% %X = img+300
% figure
% imshow(img_noisy)
%figure
%imshow(img_double)

%imshow(img_noisy)
%title(['Noisy IMG' , '\mu_n = ',num2str(0) ,'\sigma_n = ', num2str(gaussian_noise_sigma)] )

%% Shear let part
Shearlet_toolbox_path = 'C:\Users\Arian\sample\shearlet_toolbox_1'

Shearlet_folder = genpath(Shearlet_toolbox_path);
addpath(Shearlet_folder);
%SLQdecThreshRec(X,nScales,thresholdingFactors);
%brb_img =imread('barbara.jpg');
folder = fileparts(which(Shearlet_toolbox_path));

% This file gives an example of shearlet denoising.
% Code contributors: Glenn R. Easley, Demetrio Labate, and Wang-Q Lim

% Determine weather coefficients are to be displayed or not
%display_flag=0; % Do not display coefficients
display_flag=1; % Display coefficients
 %% sample toolbox denoising
 
% Load image
%x=double(imread('boat.png'));

%x=double(imread('pepper.png'));
x=double(imread('boat.png'));
[L L]=size(x);

% Create noisy image
sigma=20;
x_noisy=x+sigma.*randn(L,L);

% setup parameters for shearlet transform
lpfilt='maxflat';
% .dcomp(i) indicates there will be 2^dcomp(i) directions 
shear_parameters.dcomp =[ 2  3  4  4];
% .dsize(i) indicate the local directional filter will be
% dsize(i) by dsize(i)
shear_parameters.dsize =[32 32 16 16];
% % 
% %Tscalars determine the thresholding multipliers for
% %standard deviation noise estimates. Tscalars(1) is the
% %threshold scalar for the low-pass coefficients, Tscalars(2)
% %is the threshold scalar for the band-pass coefficients, 
% %Tscalars(3) is the threshold scalar for the high-pass
% %coefficients. 
% 
% Tscalars=[0 3 4];
% 
% %There are three possible ways of implementing the 
% %local nonsubsampled shearlet transform (nsst_dec1e,
% %nsst_dec1, nsst_dec2). For this demo, we have created 
% %a flag called shear_version to choose which one to
% %test.
% 
% shear_version=0; %nsst_dec1e
% %shear_version=1; %nsst_dec1
% %shear_version=2; %nsst_dec2
% 
% % compute the shearlet decompositon
% if shear_version==0,
%   [dst,shear_f]=nsst_dec1e(x_noisy,shear_parameters,lpfilt);
% elseif shear_version==1, 
%   [dst,shear_f]=nsst_dec1(x_noisy,shear_parameters,lpfilt);
% elseif shear_version==2
%   [dst,shear_f]=nsst_dec2(x_noisy,shear_parameters,lpfilt);
% end
% 
% % Determines via Monte Carlo the standard deviation of
% % the white Gaussian noise for each scale and 
% % directional component when a white Gaussian noise of
% % standard deviation of 1 is feed through.
% if shear_version==0,
%    dst_scalars=nsst_scalars_e(L,shear_f,lpfilt);
% else
%    dst_scalars=nsst_scalars(L,shear_f,lpfilt);
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%% display coefficients %%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% if display_flag==1,
%    figure(1)
%    imagesc(dst{1})
%    for i=1:length(dst)-1,
%        l=size(dst{i+1},3);
%        JC=ceil(l/2);
%        JR=ceil(l/JC);
%        figure(i+1)
%        sgtitle(['Level = ' , num2str(i)])
%        for k=1:l,
%            subplot(JR,JC,k)
%            
%            imagesc(abs(dst{i+1}(:,:,k)))
%            
%            axis off
%            axis image
%            title(['Direction Idx = ', num2str(k)])
%        end   
%    end
% end % display_flag 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % apply hard threshold to the shearlet coefficients
% dst=nsst_HT(dst,sigma,Tscalars,dst_scalars);
% 
% 
% % reconstruct the image from the shearlet coefficients
% if shear_version==0,
%     xr=nsst_rec1(dst,lpfilt);      
% elseif shear_version==1,
%     xr=nsst_rec1(dst,lpfilt);      
% elseif shear_version==2,
%     xr=nsst_rec2(dst,shear_f,lpfilt);      
% end
% 
%        
% % compute measures of performance
% p0 = MeanSquareError(x,x_noisy);
% fprintf('Initial MSE = %f\n',p0);
% p1 = MeanSquareError(x,xr);
% fprintf('MSE After Denoising = %f\n',p1);
% fprintf('Relative Error (norm) After Denoising = %f\n',norm(xr-x)/norm(x));
% %RECONTRUCTION_ERROR = norm(xr-x)/norm(x)
% 
% figure(10)
% imagesc(x)
% title(['ORIGINAL IMAGE, size = ',num2str(L),' x ',num2str(L)])
% colormap('gray')
% axis off
% axis image
% 
% figure(11)
% imagesc(x_noisy)
% title(['NOISY IMAGE, MSE = ',num2str(p0)])
% colormap('gray')
% axis off
% axis image
% 
% figure(12)
% imagesc(xr)
% title(['RESTORED IMAGE, MSE = ',num2str(p1)])
% colormap('gray')
% axis off
% axis image

%% my main
[dst,shear_f]=nsst_dec2(x_noisy,shear_parameters,lpfilt);
%creat_X_dataset(dst, 1,2,3)
%plot_hist_shearlet_coeffs(dst,'ShearletCoeffPepper','D:\MSC\Term4\Thesis\Report\Boat');
%plot_hist_shearlet_coeffs(dst,'ShearletCoeffPepper','D:\MSC\Term4\Thesis\Report\Pepper');
XX=creat_X_dataset(dst, 2,1,3);
rowData_X = XX(1,:);
% [f1,x1] = ecdf(XX(1,:));
% [f2,x2] = ecdf(XX(2,:));
% [f3,x3] = ecdf(XX(3,:));

%rhohat = copulafit('Gaussian',[f1( f1~=1 & f1~=0),f1( f1~=1 & f1~=0),f1( f1~=1 & f1~=0)])
%u = ksdensity(XX(1,:),XX(1,:),'function','cdf');
%v = ksdensity(XX(2,:),XX(2,:),'function','cdf');
%rhohat = copulafit('Gaussian',[u,v])

%cdfplot(XX(1,:))
%hist3(XX([2,1],:)','Nbins',[50,50])
%histfit(XX(1,:),100,'normal')
%XX=creat_X_dataset2(dst, 1,1,2);
% [cA,cH,cV,cD] = dwt2(x,'sym4','mode','per');
% imshow(cD)
% %hist(cD(:),100)
% XX = creat_X_dataset_wave(cA,3)
% histogram2(XX(1,:),XX(2,:),100)
