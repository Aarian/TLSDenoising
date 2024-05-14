function denoised_sacale_level = scale_level_denoising_vDataset(Dataset_XX_T, RVn_lst,JointPrior, Sigma_n)
%% Parameters 
    % Path_XX : path to data set XX 
    % lst_RVn : row vector of selected columns 
    % Joint Prior : Joint Prior using copula
    % Sigma_n : Sigma noise 
%% return : 
    % denoised scale level as SQ matrix 
    RVn = size(RVn_lst,2) ;
    
    %Dataset_XX_T = load (path_XX); 
    Dataset_XX  =Dataset_XX_T' ; 
    sz_x= size (Dataset_XX,1);
    [up,down]=  MMSE(JointPrior,RVn,Sigma_n,RVn);
    denoised_store = zeros (512,512) ; 
    for i = 1: sz_x 
        idx_dataset = i;
        idx_denoised = [Dataset_XX(i,end-1),Dataset_XX(i,end)]
        observed = Dataset_XX(i,RVn_lst)
        
        estimated = integral_MMSE(up, down,observed',RVn) 
        denoised_store (Dataset_XX(i,end-1),Dataset_XX(i,end)) = estimated ; 
        
    end
    denoised_sacale_level = denoised_store ; 


end