SAR_image_path = 'vie1_0R.jpg';
SAR_image_path = 'vie1_0R.jpg';
Sigma_noise = 0 ; % it must estimated from noisy img
    my_img_sz = 512
    %img_clean=double(imread('boat.png'));
    
    img_clean=double(rgb2gray(imread(SAR_image_path)));
    %img_clean = double(imread(SAR_image_path)) ; 
    
       my_img_sz = 512
    %img_clean=double(imread('boat.png'));

    
    img_clean = imresize(img_clean,[my_img_sz my_img_sz]); 
    sigma = 0.2 ;
         %%%% LOG VERSION
    [L L]=size(img_clean);
    x = log(img_clean+1); 
    x_noisy=x+sigma.*randn(L,L);
    
X_0R = x_noisy;
X_1R = imrotate(x_noisy,+90,'bilinear','crop');
X_2R = imrotate(x_noisy,+180,'bilinear','crop');
X_3R = imrotate(x_noisy,+270,'bilinear','crop');   
    

[noise_info,res_table,Info_line,xr_0R] = resulter_new (exp(X_0R) , 'vie1_0R.jpg', Sigma_noise,[1], [1 2 2], 'BCGM');
[noise_info,res_table,Info_line,xr_1R] = resulter_new (exp(X_1R) ,'vie1_1R.jpg', Sigma_noise,[1], [1 2 2], 'BCGM');
[noise_info,res_table,Info_line,xr_2R] = resulter_new (exp(X_2R) ,'vie1_2R.jpg', Sigma_noise,[1], [1 2 2], 'BCGM');
[noise_info,res_table,Info_line,xr_3R] = resulter_new (exp(X_3R) ,'vie1_3R.jpg', Sigma_noise,[1], [1 2 2], 'BCGM');

% 
% [noise_info,res_table,Info_line,xr_0R] = resulter_Davoodi ('SAR2_0R.png', Sigma_noise,[1], [1 2 2], 'BCGM');
% [noise_info,res_table,Info_line,xr_1R] = resulter_Davoodi ('SAR2_1R.png', Sigma_noise,[1], [1 2 2], 'BCGM');
% [noise_info,res_table,Info_line,xr_2R] = resulter_Davoodi ('SAR2_2R.png', Sigma_noise,[1], [1 2 2], 'BCGM');
% [noise_info,res_table,Info_line,xr_3R] = resulter_Davoodi ('SAR2_3R.png', Sigma_noise,[1], [1 2 2], 'BCGM');
% imshow(imrotate(xr_1R,-90,'bilinear','crop'),[])
% imshow(imrotate(xr_2R,-90,'bilinear','crop'),[])
% imshow(imrotate(xr_3R,-180,'bilinear','crop'),[])

J_0R = xr_0R;
 J_1R = imrotate(xr_1R,-90,'bilinear','crop');
J_2R = imrotate(xr_2R,-180,'bilinear','crop');
J_3R = imrotate(xr_3R,-270,'bilinear','crop');

 JJ = uint8(.25*(exp(double(J_3R))+exp(double(J_2R))+exp(double(J_1R))+exp(double(J_0R)))) ;
imshow(JJ,[])

%     my_img_sz = 512
%     %img_clean=double(imread('boat.png'));
%     img_clean=double(rgb2gray(imread(SAR_image_path)));
%     img_clean = imresize(img_clean,[my_img_sz my_img_sz]);
%         diff_0  = JJ - J_0R  ;
%         diff_1  = JJ - J_1R  ;
%         diff_2  = JJ - J_2R  ;
%         diff_3  = JJ - J_3R  ;
%     out_put = zeros(512,512) ;
%    for ii = 1:my_img_sz 
%      for jj = 1: my_img_sz
%             [~,pos] = min(abs([diff_0(ii,jj),diff_1(ii,jj),diff_2(ii,jj),diff_3(ii,jj)])); 
%             if(pos == 1)
%                 out_put(ii,jj) = J_0R(ii,jj) ;
%             end
%                         if(pos == 2)
%                 out_put(ii,jj) = J_1R(ii,jj) ;
%                         end
%                         if(pos == 3)
%                 out_put(ii,jj) = J_2R(ii,jj) ;
%                         end
%                         if(pos == 4)
%                 out_put(ii,jj) = J_3R(ii,jj) ;
%             end
%      end
%  end
%     figure 
%     title('novel min')
%     imshow(uint8(out_put),[]) ; 
    
    immse((JJ),  uint8(img_clean))
        psnr((JJ),uint8(img_clean))
        ssim((JJ),uint8(img_clean))
        epi(double(JJ),(img_clean))