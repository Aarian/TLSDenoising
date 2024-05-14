function denoised_sacale_level = scale_level_conv_denoising(XX, RVn_lst,JointPrior, Sigma_n,xmin,xmax,integral_bins ,L,gpu_flag)
%% Parameters 
    % Path_XX : path to data set XX 
    % lst_RVn : row vector of selected columns 
    % Joint Prior : Joint Prior using copula
    % Sigma_n : Sigma noise 
%% return : 
    % denoised scale level as SQ matrix 
    RVn = size(RVn_lst,2) ;
    %xmin = min ( XX(1,:))
    %xmax = max ( XX(1,:))
    Dataset_XX_T =  (XX); 
    Dataset_XX  =Dataset_XX_T' ; 
    sz_x= size (Dataset_XX,1);
    %[up,down]=  MMSE(JointPrior,RVn,Sigma_n,RVn);
    P_noise_sym = P_noise( Sigma_n, RVn); 
    [Samples MMSE_res] = Bayes_core(P_noise_sym ,JointPrior,xmin,xmax,integral_bins,RVn,gpu_flag) ; 
    denoised_store = zeros (L,L) ; 
    for i = 1: sz_x 
        idx_dataset = i;
        idx_denoised = [Dataset_XX(i,end-1),Dataset_XX(i,end)];
        observed = Dataset_XX(i,RVn_lst);
        if (RVn ==1 )
            %Samples
            X = Samples.x;
            sz_MMSE_res = size(MMSE_res);
            estimated = interpn(X,MMSE_res,observed(1),'LINEAR',-1);   
            %estimated = interp1(X,MMSE_res,observed(1),'Linear','extrap');
        end
        if (RVn ==2 )
            %Samples
            X = Samples.x;
            Y = Samples.y ;
            sz_MMSE_res = size(MMSE_res);
            estimated = interpn(X,Y,MMSE_res,observed(1),observed(2),'LINEAR',-1);  
            %estimated = interp2(X,Y,MMSE_res,observed(1),observed(2),'Linear',-1);
        end
        if (RVn ==3 )
            X = Samples.x;
            Y = Samples.y ;
            Z = Samples.z ;
            %estimated = interpn(X,Y,Z,MMSE_res,observed(1),observed(2), observed(3),'Linear');
            estimated = interp3(X,Y,Z,MMSE_res,observed(1),observed(2), observed(3),'Linear',-1);
            %estimated = interp3(X,Y,Z,MMSE_res,observed(1),observed(2), observed(3),'spline',0);
        end        
        %estimated = integral_MMSE(up, down,observed',RVn,gpu_flag); 
        denoised_store (Dataset_XX(i,end-1),Dataset_XX(i,end)) = gather(estimated) ; 
        
    end
    denoised_sacale_level = denoised_store  ;


end