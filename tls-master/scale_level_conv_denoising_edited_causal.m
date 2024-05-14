function denoised_sacale_level = scale_level_conv_denoising_edited_causal(XX, RVn_lst,JointPrior, Sigma_n,xmin,xmax,integral_bins ,L,gpu_flag)
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
    coeff_integ_IDX = 1:RVn ;
    cell_samples ={} ;
    cell_MMSE ={} ;
    for i = 1:RVn
        [Samples MMSE_res] = Bayes_core_edited(P_noise_sym ,JointPrior,coeff_integ_IDX(i),xmin,xmax,integral_bins,RVn,gpu_flag) ; 
        %[Samples MMSE_res] = Bayes_core_gaussian(P_noise_sym ,JointPrior,coeff_integ_IDX(i),Sigma_n,xmin,xmax,integral_bins,RVn,gpu_flag); 
        cell_samples{i} =Samples;
        cell_MMSE{i} = MMSE_res ;
    end
    %[Samples MMSE_res] = Bayes_core_edited(P_noise_sym ,JointPrior,coeff_integ_IDX(1),xmin,xmax,integral_bins,RVn,gpu_flag) ; 
    %[Samples2 MMSE_res2] = Bayes_core_edited(P_noise_sym ,JointPrior,coeff_integ_IDX(2),xmin,xmax,integral_bins,RVn,gpu_flag) ; 
    %cell_samples ={};
    %cell_MMSE = {} ;
    denoised_store = zeros (L,L) ; 
    for i = 1:sz_x 
        idx_dataset = i;
        idx_denoised = [Dataset_XX(i,end-1),Dataset_XX(i,end)];
        ii = Dataset_XX(i,end-1) ;  
        jj = Dataset_XX(i,end);
        %observed = Dataset_XX(floor((i-1)/RVn)+1,RVn_lst);
        if(ii<3 || jj < 3)
            observed = Dataset_XX(i,RVn_lst);
        else
           tmp_row = Dataset_XX(i,:);
           obs_1_ij = tmp_row (1) ;
           
           tmp_row = Dataset_XX(L*(ii-2-1)+(jj-2),:);
           obs_2_i2j2 = tmp_row (2) ;
           
           tmp_row = Dataset_XX(L*(ii-2-1)+jj,:);
           obs_3_i2j = tmp_row (3) ;
           
           tmp_row = Dataset_XX(L*(ii-1) + (jj-2),:);
           obs_4_ij2 = tmp_row (4) ;
           
           row_causal = [obs_1_ij , obs_2_i2j2, obs_3_i2j  ,obs_4_ij2 ]  ; 
           
           observed = row_causal (1,RVn_lst) ;
        end
        if (RVn ==1 )
            %Samples
            X = Samples.x;
            
            sz_MMSE_res = size(MMSE_res);
            if (mod(i,RVn) == 0)
                estimated = interp1(X,cell_MMSE{RVn},observed(1),'Linear','extrap');
            else
                estimated = interp1(X,cell_MMSE{mod(i,RVn)},observed(1),'Linear','extrap');    
            end
        end
        if (RVn ==2 )
            %Samples
            X = Samples.x;
            Y = Samples.y ;
            sz_MMSE_res = size(MMSE_res);
            if (mod(i,RVn) == 0)
                estimated = interp2(X,Y,cell_MMSE{RVn},observed(1),observed(2),'Linear',-1);
                %estimated = interp2(X,Y,cell_MMSE{RVn},observed(1),observed(2),'spline');
            else
                estimated = interp2(X,Y,cell_MMSE{mod(i,RVn)},observed(1),observed(2),'Linear',-1);   
                %estimated = interp2(X,Y,cell_MMSE{mod(i,RVn)},observed(1),observed(2),'spline'); 
            end
        end
        if (RVn ==3 )
            X = Samples.x;
            Y = Samples.y ;
            Z = Samples.z ;
            mod(i,RVn);
            if (mod(i,RVn) == 0)
                estimated = interp3(X,Y,Z,cell_MMSE{RVn},observed(1),observed(2), observed(3),'spline',0);
                %estimated = interp3(X,Y,Z,cell_MMSE{RVn},observed(1),observed(2), observed(3),'Linear',-1);
            else
                estimated = interp3(X,Y,Z,cell_MMSE{mod(i,RVn)},observed(1),observed(2), observed(3),'spline',0);
                %estimated = interp3(X,Y,Z,cell_MMSE{RVn},observed(1),observed(2), observed(3),'Linear',-1);
            end
            
        end        
        %estimated = integral_MMSE(up, down,observed',RVn,gpu_flag); 
        if (gpu_flag==1)
            denoised_store (Dataset_XX(i,end-1),Dataset_XX(i,end)) = gather(estimated) ;
        else
            denoised_store (Dataset_XX(i,end-1),Dataset_XX(i,end)) = (estimated) ;
            if(i >= 2*L)
                Dataset_XX(i,1) = estimated  ; 
                Dataset_XX(i-L-1,2) = estimated  ; 
                Dataset_XX(i-1,3) = estimated  ; 
                Dataset_XX(i-L,4) = estimated  ;
            end
        end
        
    end
    denoised_sacale_level = denoised_store  ;


end