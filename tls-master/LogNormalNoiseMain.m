

 %% sample toolbox denoising
 
% Load image
%x=double(imread('boat.png'));

%x=double(imread('pepper.png'));
my_img_sz = 512 ;
%img_clean=double(imread('boat.png'));
img_clean=double(rgb2gray(imread('vie1_0R.jpg')));
img_clean = imresize(img_clean,[my_img_sz my_img_sz]);
[L,L] = size(img_clean); 
mu = 0 ;
enl =1 ;
sigma = .4 ;

M = exp(mu + .5*sigma^2);

MNoise = lognrnd(mu,sigma, [L,L]) ; 
log_normal_achim = exp(sqrt(2*log(M/enl)).*(sigma.*randn(L,L)) + log(enl));
Noisy_img1 = img_clean .* log_normal_achim ;


P_gamma = gamrnd(enl , enl , L,L) ;
%Noisy_img1 = P_gamma .* img_clean ; 

imshow(uint8(Noisy_img1),[])
% addpath('D:\MSC\Term4\Thesis\GitBackupThesis\thesis\sample\ThesisCode\Achim') 
% 
% x = log(Noisy_img1);
% 
% J = 5;
% I=sqrt(-1);
% 
% % symmetric extension
% L = length(x); % length of the original image.
% N = L+2^J;     % length after extension.
% x = symextend(x,2^(J-1));
% 
% [Faf, Fsf] = AntonB;
% [af, sf] = dualfilt1;
% W = cplxdual2D(x, J, Faf, af);
% W = normcoef(W,J,nor);
% 
% % Noise variance estimation using robust median estimator..
% tmp = W{1}{1}{1}{3};
% Nsig = median(abs(tmp(:)))/0.6745

Noisy_img = img_clean .* log_normal_achim ;
%imagesc(Noisy_img)
% figure ()
% 
% J = imnoise(img_clean,'speckle',.001)
% imshow(J, [])
% title('using title noise')
% 
% figure()
% 
% subplot(1,2,1)
% imshow(uint8(Noisy_img), [])
% title('achim')
% subplot(1,2,2)
% imshow(uint8(Noisy_img1), [])
% title('matlab')
% figure
% 
% subplot(1,2,1)
%  hist(MNoise(:),100)
%   title('matlab')
% 
% subplot(1,2,2)
% 
%  hist(log_normal_achim(:),100)
%      title('achim')