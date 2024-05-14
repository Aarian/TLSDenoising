
%% Shear let part
Shearlet_toolbox_path = 'C:\Users\Arian\GitBackupThesis\thesis\sample\shearlet_toolbox_1';
Stable_path = 'C:\Users\Arian\GitBackupThesis\thesis\stbl-master';
Shearlet_folder = genpath(Shearlet_toolbox_path);
addpath(Shearlet_folder);
addpath (genpath(Stable_path));

%SLQdecThreshRec(X,nScales,thresholdingFactors);
%brb_img =imread('barbara.jpg');
folder = fileparts(which(Shearlet_toolbox_path));

 %% sample toolbox denoising
 
% Load image
%x=double(imread('boat.png'));

%x=double(imread('pepper.png'));
my_img_sz = 512
%img_clean=double(imread('boat.png'));
img_clean=double(rgb2gray(imread('kit1.jpg')));
img_clean = imresize(img_clean,[my_img_sz my_img_sz]);

%img_clean=double(rgb2gray(imread('aerial2.png')));
%img_clean = imresize(img_clean,[my_img_sz my_img_sz]);
%img_clean=double(imread('p07_003.png'));
%x = imresize(x,[my_img_sz,my_img_sz]);
%figure()
%imagesc(log(x))
x =img_clean;
[L L]=size(x)
%imshow(x)
% Create noisy image

 sigma=0.2;
 Sigma_n = sigma ;
 %%%% LOG VERSION
x = log(x+1); 
 x_noisy=x+sigma.*randn(L,L);
%figure()
%imagesc(x_noisy)

%%%%% WITHOUT LOG
%x_noisy=x+sigma.*randn(L,L);
%RVn_lst = [1,2,4]
%RVn_lst = [1,2]
RVn_lst = [1]
RVn = size(RVn_lst,2);
gpu_flg = 0 ;
% setup parameters for shearlet transform
lpfilt='maxflat';
% .dcomp(i) indicates there will be 2^dcomp(i) directions 
decomp = [1 2 2 ]

shear_parameters.dcomp =decomp;
% .dsize(i) indicate the local directional filter will be
% dsize(i) by dsize(i)

shear_parameters.dsize =[32 32 16 16];
[dst_clean,shear_f_clean]=nsst_dec1e(x,shear_parameters,lpfilt);
[dst,shear_f]=nsst_dec1e(x_noisy,shear_parameters,lpfilt);
dst_scalars_noise_level = Noise_Estimation(L,shear_f,lpfilt,sigma);
my_dst{1} = dst{1}; 
for i = 1:length(decomp) 
    for j = 1:2^decomp(i)
        level = i+1 ;
        dir = j;
        Sigma_n = dst_scalars_noise_level{level}(dir)
        
        XX_scale_level_mtb = creat_X_dataset(dst, level, dir, 3);
        XX_scale_level_mtb_clean = creat_X_dataset(dst_clean, level, dir, 3);
        coeff = dst{level}(:,:,dir);
        %xmin = min ( XX_scale_level_mtb(1,:));
        %xmax = max (  XX_scale_level_mtb(1,:));
        xmin = min(coeff(:))
        xmax = max(coeff(:))
        integral_bins =10;
        XX = XX_scale_level_mtb(RVn_lst, :);
        fea = XX_scale_level_mtb(RVn_lst, 1:20:end)';
        XX_clean = XX_scale_level_mtb_clean(RVn_lst, :);
        %Sigma_copula = copulafit('Gaussian',  )
        %x = XX(1,1:20:end)';
        %y = XX(2,1:20:end)';
        %fea1 = XX_clean(1,1:20:end)';
        %fea2 = XX_clean(2,1:20:end)';
        %z = XX(3,1:20:end)'; 
        %scatterhist(x,y)
        %u = ksdensity(x,x,'function', 'cdf');
        %v = ksdensity(y,y,'function', 'cdf');
        %u = ksdensity(fea1,fea1,'function', 'cdf');
        %v = ksdensity(fea2,fea2,'function', 'cdf');
        %k = ksdensity(z,z,'function', 'cdf');
        %scatterhist(u,v)
        %R= corrcoef(u,v)
        %Sigma_copula = copulafit('Gaussian', [u,v])
        Sigma_copula = copula_estimator (fea) ;
        %Sigma_copula = copulafit('Gaussian', [u,v,k])
        %Sigma_copula = [1]
        %Sigma_copula = eye(RVn) % independent !
        
        %% BCGM
        
        p = stblfit(fea(:,1))
        scale = p(3);
        alpha = p(1);
        eps = (4-alpha^2) / (3*alpha^2)
        JointPrior = Copula_multivariate(BCGM_pdf(eps,scale),Sigma_copula,RVn);
        %% TLS
%         pd = fitdist(fea1,'tLocationScale')
%         JointPrior = Copula_multivariate(tls(pd.mu,pd.sigma,pd.nu),Sigma_copula,RVn);
%         Joint = matlabFunction (JointPrior)
        %Joint(1,2,3)
        %cdf = int (matlabFunction( simplify(tls(pd.mu,pd.sigma,pd.nu))),-inf,sym('x1'))
       
        %denoised_sacale_level = scale_level_denoising(XX_scale_level_mtb,RVn_lst,JointPrior, Sigma_n,L,gpu_flg);
        %denoised_sacale_level = scale_level_conv_denoising(XX_scale_level_mtb, RVn_lst,JointPrior, Sigma_n,xmin,xmax,integral_bins ,L,gpu_flg);
        
        
        
        denoised_sacale_level = scale_level_conv_denoising_edited(XX_scale_level_mtb, RVn_lst,JointPrior, Sigma_n,xmin,xmax,integral_bins ,L,gpu_flg);
        %dst{level}(:,:,dir) = denoised_sacale_level ;
        my_dst{level}(:,:,dir) = denoised_sacale_level ;
        %figure()
        %imagesc(denoised_sacale_level)
    end
end
%xr=nsst_rec1(dst,lpfilt)
xr=nsst_rec1(my_dst,lpfilt);
%imshow(exp(xr))
figure()

imshow(uint8(exp(xr)),[])
title('uint8 xr')
figure()

imshow(uint8(exp(x_noisy)),[])
title ('uint8 x_noisy')
figure()

imagesc(xr)
title ('xr')
figure()

imagesc(x_noisy)
title ('x_noisy')

figure()
imshow(uint8(img_clean),[])
title ('image clean')

 
img_lee = myLee(uint8(exp(xr))) ; 
img_lee = uint8(img_lee) ;
f_lee = figure() ;
imshow(img_lee,[]);
title('Filtered xr by Lee');
%saveas(f_lee,fullfile(convertStringsToChars(saving_path),[convertStringsToChars(img_name),'Lee.png']));


RVn_lst;

MSE_clean_denoisaed = immse(uint8(exp(xr)),  uint8(img_clean)) 
MSE_xrlee_clean = immse(uint8(img_lee),  uint8(img_clean)) 
MSE_clean_noisy = immse(uint8(exp(x_noisy)),  uint8(img_clean)) 
