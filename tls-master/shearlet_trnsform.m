function [dst,shear_f] = shearlet_trnsform (img, sh_dcomp, sh_dsize, lpfilter)
    %% parameter 
        % img : image 
        % sh_dcomp : shearlet decomp levels row vector .dcomp(i) indicates there will be 2^dcomp(i) directions 
        % sh_dsize : .dsize(i) indicate the local directional filter will
        % be row vector
        % lpfilter : str name lpfilter
   %% return 
    % dst : shear let coeff (cell array)

    
    shear_parameters.dcomp = sh_dcomp;
    shear_parameters.dsize = sh_dsize;
    lpfilt=lpfilter ; 
    %[dst,shear_f]=nsst_dec2(img,shear_parameters,lpfilt);
    [dst,shear_f]=nsst_dec1e(img,shear_parameters,lpfilt);
end