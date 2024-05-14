function [xr_exp,dst,dst_new] = hard_treshold(img,sigma_n)
    % you need to import SHEARLET TOOLBOX for this function 
    if sigma_n == 0 
        Sigma_noise = estimate_unknown_sigma_noise(img) ; 
    else
        Sigma_noise = sigma_n ; 
    end
    [L,L] = size (img);
    lpfilt='maxflat';
    shear_parameters.dcomp =[ 1 2 2];

    shear_parameters.dsize =[32 32 16 16];
    x_noisy = log(img+1) ; 
    %x_noisy =  img ;
    Tscalars=[0 5 2.5];
    %Tscalars=[0 15 7.5]; % for the real version
    [dst,shear_f]=nsst_dec1e(x_noisy,shear_parameters,lpfilt);
    dst_scalars =Noise_Estimation(L,shear_f,lpfilt,Sigma_noise) ;
    dst_new=nsst_HT(dst,Sigma_noise,Tscalars,dst_scalars);
    xr=nsst_rec1(dst_new,lpfilt);
    xr_exp = (exp(xr)) ; 
end