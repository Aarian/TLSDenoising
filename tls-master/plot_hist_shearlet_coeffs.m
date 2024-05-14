function [] = plot_hist_shearlet_coeffs(coeffs_img,Title,Folder_path)
    % Inputs:
%
% coeff_img                       - cell array contain all the shearlet coeffs of img
% Title                           - Name of the image
% Folder Name                     - Folder Name
% Output:
%
% []                    - 
%
% 
% Copyright 2019 by Arian Morteza. All Rights Reserved.
%
   nbins=100;
   Coeff_folder = genpath(Folder_path);
   addpath(Coeff_folder);
   mkdir(Folder_path);
   fig_hist = figure(1);
   %imagesc(coeffs_img{1})
   samples_Ncoeff = creat_X_dataset(coeffs_img,1,1,3); 
   hist(samples_Ncoeff (1,:),nbins) ;
   saveas(fig_hist,fullfile(Folder_path,[Title,'Hist', '_',num2str(1),'.eps']));
   
   fig_coeff = figure(2) 
   imagesc(abs(coeffs_img{1}(:,:)));
   title([Title , ' Level = 0'])
   
   saveas(fig_coeff,fullfile(Folder_path,[Title, '_',num2str(1),'.eps']));
   for i=1:length(coeffs_img)-1,
       l=size(coeffs_img{i+1},3);
       JC=ceil(l/2);
       JR=ceil(l/JC);
       fig_hist = figure(i+1);
       
       sgtitle([Title , '  Level = ' , num2str(i)])
       for k=1:l,
           subplot(JR,JC,k)
           samples_Ncoefff = creat_X_dataset(coeffs_img,i+1,k,3);
           hist(samples_Ncoefff(1,:),nbins) ;

           title(['Direction Idx = ', num2str(k)])
       end 
       saveas(fig_hist,fullfile(Folder_path,[Title,'Hist', '_',num2str(i+1),'.eps']));
       
       fig_coeff = figure(i+1);
       sgtitle([Title , '  Level = ' , num2str(i)])
       for k=1:l,
           subplot(JR,JC,k) ;
           imagesc(abs(coeffs_img{i+1}(:,:,k)));
           title(['Direction Idx = ', num2str(k)])
       end 
       %% saving file
       
       saveas(fig_coeff,fullfile(Folder_path,[Title, '_',num2str(i+1),'.eps']));
   end % end for
end% function