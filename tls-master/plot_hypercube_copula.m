function [noise_info,res_table,Info_line,x_rec,dst,my_dst,dst_clean,TLS_table] =  plot_hypercube_copula (SAR_image_path, Sigma_noise,RandomVariables_lst, Shearlet_decomp_lst, method_name)
    close all
    %T = table({''},[.0],[.0],[.0],{''},[Sigma_noise],[RandomVariables_lst],{'Unkonwn'},'VariableNames',{'ImgName','MSE','PSNR', 'EPI', 'PriorDist','Sigma_noise','RV_lst', 'dependency'} ) 
    saving_path = "C:\Users\arian\Desktop\ARR\Resualts\";
    saving_path_CopulaHyper = 'C:\Users\arian\Desktop\ARR\tls-master\Results_Copula_HyperCube'
    %% Shear let part
    Shearlet_toolbox_path = 'C:\Users\arian\Desktop\ARR\shearlet_toolbox_1';
    Stable_path = 'C:\Users\arian\Desktop\ARR\stbl-master';
    Achim_path = 'C:\Users\arian\Desktop\ARR\Achim' ; 
    CauchyD_path = 'C:\Users\arian\Desktop\ARR\CauchyD' ; 
    cs_nsst_path = 'C:\Users\arian\Desktop\ARR\cs_nsst';
    SaS_path = 'C:\Users\arian\Desktop\ARR\SaSDenoising';
    Shearlet_folder = genpath(Shearlet_toolbox_path);
    addpath(Shearlet_folder);
    addpath(CauchyD_path);
    addpath(genpath(cs_nsst_path)) ; 
    addpath (genpath(Stable_path));
    addpath (genpath(SaS_path));
    addpath (genpath(Achim_path));
    %SLQdecThreshRec(X,nScales,thresholdingFactors);
    %brb_img =imread('barbara.jpg');
    folder = fileparts(which(Shearlet_toolbox_path));

     %% sample toolbox denoising

    % Load image
    
    

    
    
    %x=double(imread('boat.png'));

    %x=double(imread('pepper.png'));
    my_img_sz = 512 %% for synthetic
     %my_img_sz = 160
    img_clean=double(imread(SAR_image_path));
    %img_clean=double(rgb2gray(imread(SAR_image_path)));
    img_clean = imresize(img_clean,[my_img_sz my_img_sz]);
    %img_clean=double(rgb2gray(imread('aerial2.png')));
    
    %img_clean = imresize(img_clean,[my_img_sz my_img_sz]);
    %img_clean=double(imread('p07_003.png'));
    %x = imresize(x,[my_img_sz,my_img_sz]);
    %figure()
    %imagesc(log(x))
%      f4 =figure()
%     imshow(uint8(img_clean),[])
    
    %imshow(x)
    % Create noisy image
    if(Sigma_noise == 0)
        disp('original version')
         x =img_clean;
        [L,L] = size(img_clean);
        x_exp = img_clean;
        x_noisy = log(x_exp+1);
        sigma = estimate_unknown_sigma_noise(x_exp);
    else
        disp('synthetic version')
        x =img_clean;
        [L L]=size(x)

            [L,L] = size(img_clean); 
        mu = 0 ;
        enl =1 ;
        sigma = Sigma_noise ;

        M = exp(mu + .5*sigma^2);
        log_normal_achim = exp(sqrt(2*log(M/enl)).*(sigma.*randn(L,L)) + log(enl));
        x_exp = img_clean .* log_normal_achim ;
        x_noisy = log(x_exp+1); 
         sigma=Sigma_noise;
    end
    
    
     Sigma_n = sigma ;
     %% LOG VERSION
    %x = log(x+1); 
    %x_noisy=x+sigma.*randn(L,L);
    %figure()
    %imagesc(x_noisy)

    %%%%% WITHOUT LOG
    %x_noisy=x+sigma.*randn(L,L);
    %RVn_lst = [1,2,4]
    RVn_lst = RandomVariables_lst
    %RVn_lst = [1]
    RVn = size(RVn_lst,2);
    gpu_flg = 0 ;
    % setup parameters for shearlet transform
    lpfilt='maxflat';
    % .dcomp(i) indicates there will be 2^dcomp(i) directions 
    decomp = Shearlet_decomp_lst

    shear_parameters.dcomp =decomp;
    % .dsize(i) indicate the local directional filter will be
    % dsize(i) by dsize(i)

    shear_parameters.dsize =[32 32 16 16];
    [dst_clean,shear_f_clean]=nsst_dec1e(log(x+1),shear_parameters,lpfilt);
    [dst,shear_f]=nsst_dec1e(x_noisy,shear_parameters,lpfilt);
%     kptkpt = dst_clean{4}(:,:,1);
%     kptkpt_max = max(kptkpt(:))  
%     kptkpt_min = min(kptkpt(:)) 
    dst_scalars_noise_level = Noise_Estimation(L,shear_f,lpfilt,sigma);
    my_dst{1} = dst{1}; 
    
    img_wiener = wiener2(uint8(exp(x_noisy)),[5 5]);
    img_frost = fcnFrostFilter(uint8(exp(x_noisy)));
    img_lee = myLee(uint8(exp(x_noisy))) ; 
    img_lee = uint8(img_lee) ;
    
    [dst_lee,shear_f_lee]=nsst_dec1e(log(double(img_lee)+1),shear_parameters,lpfilt);
    
    
    splited_path = split(string(SAR_image_path),"\") 
    img_name_lst = split(splited_path(end),".") 
    img_name = img_name_lst(1) 
    
        f2=figure();
    imshow(uint8(exp(x_noisy)))
    title ('image noisy')
    
    
        f4 =figure()
    imshow(uint8(img_clean),[])
    title ('Clean Image')
    %Image = getframe(gcf);
    Filename = join([img_name,string(method_name),"imgclean"],"_") ;
    Filename = join ([Filename ,string(num2str(size(RandomVariables_lst,2))) ] , "_") ;
    Filename = join([Filename, "png"], ".");
    %imwrite(Image.cdata, join([saving_path, Filename,size(RandomVariables_lst,2), ".png"],"") );
    saveas(f4,fullfile(convertStringsToChars(saving_path),convertStringsToChars(Filename)));
 
    
    
    
    
    f_wiener = figure() ;
    imshow(img_wiener,[]);
    title('filtered by Wiener');
    saveas(f_wiener,fullfile(convertStringsToChars(saving_path),[convertStringsToChars(img_name),'Wiener.png']));

    f_frost = figure() ;
    imshow(img_frost,[]);
    title('Filtered by Frost');
    saveas(f_frost,fullfile(convertStringsToChars(saving_path),[convertStringsToChars(img_name),'Frost.png']));
   
    
    f_lee = figure() ;
    imshow(img_lee,[]);
    title('Filtered by Lee');
    saveas(f_lee,fullfile(convertStringsToChars(saving_path),[convertStringsToChars(img_name),'Lee.png']));
    
    
    MSE_Wiener = immse(img_wiener,  uint8(img_clean))
    MSE_Frost = immse(img_frost,  uint8(img_clean)) 
    MSE_Lee = immse(uint8(img_lee),  uint8(img_clean)) 
    
    ENL = @(imgg) mean(double(imgg(:)))^2/var(double(imgg(:))) ; 
    
    line_wiener = {splited_path(end),MSE_Wiener, psnr(img_wiener,uint8(img_clean)),ssim(img_wiener,uint8(img_clean)), epi(double(img_wiener),(img_clean)),ENL(img_wiener),'Wiener',Sigma_noise,[[nan nan nan]] , "X" } ;
    line_Frost = {splited_path(end),MSE_Frost, psnr(img_frost,uint8(img_clean)), ssim(img_frost,uint8(img_clean)), epi(double(img_frost),(img_clean)),ENL(img_frost),'Frost',Sigma_noise,[[nan nan nan]] , "X" } ;
    line_Lee = {splited_path(end),MSE_Lee, psnr(img_lee,uint8(img_clean)),ssim(img_lee,uint8(img_clean)), epi(double(img_lee),(img_clean)),ENL(img_lee),'Lee',Sigma_noise,[[nan nan nan]] , "X" } ;
    
    
    MSE_clean_noisy = immse(uint8(exp(x_noisy)),  uint8(img_clean)) ;
    
    NoiseInfo_table = table({'Noisy IMG'},[MSE_clean_noisy ],[psnr(uint8(exp(x_noisy)),uint8(img_clean))],[ssim(uint8(exp(x_noisy)),uint8(img_clean))],[epi(uint8(exp(x_noisy)),uint8(img_clean))],ENL(x_noisy),{''},[Sigma_noise],[nan nan nan],{'Unkonwn'},...
        'VariableNames',{'ImgName','MSE','PSNR','SSIM', 'EPI','ENL', 'PriorDist','Sigma_noise','RV_lst', 'dependency'} ) ;
    NoiseInfo_table = [NoiseInfo_table ;line_wiener ;line_Frost; line_Lee] ;
    
    TLS_statistics_table = table ([.0],[.0],[.0],{''},[.0],{''},[.0],{''},[.0],[.0],[.0],'VariableNames',{'Scale', 'Direction', 'mu_tls','mu_ConfInterval' , 'sigma_tls','sigma_ConfInterval' , 'nu_tls','nu_ConfInterval', 'Pvalue', 'stat_test_res', 'sigma_n'}) ;
    line_TLS_statistics = {0,0,0,'',0,'',0,'',0,0,0};
    
    for i = 1:length(decomp) 
        for j = 1:2^decomp(i)
            level = i+1 ;
            dir = j;
            Sigma_n = dst_scalars_noise_level{level}(dir)

            XX_scale_level_mtb = creat_X_dataset(dst, level, dir, 3);
            XX_scale_level_mtb_clean = creat_X_dataset(dst_clean, level, dir, 3);
            XX_scale_level_Lee = creat_X_dataset(dst_lee, level, dir, 3);
            coeff = dst{level}(:,:,dir);
            %xmin = min ( XX_scale_level_mtb(1,:));
            %xmax = max (  XX_scale_level_mtb(1,:));
            xmin = min(coeff(:))
            xmax = max(coeff(:))
            %xmax = abs(xmax)
            integral_bins =10;
            XX = XX_scale_level_mtb(RVn_lst, :);
            fea = XX_scale_level_mtb(RVn_lst, 1:20:end)';
            XX_clean = XX_scale_level_mtb_clean(RVn_lst, :);
            fea_clean = XX_scale_level_mtb_clean(RVn_lst, 1:20:end)';
            fea_Lee = XX_scale_level_Lee(RVn_lst, 1:20:end)';
            
            %fea = XX_scale_level_mtb_clean(RVn_lst, 1:20:end)'; 
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
            %Sigma_copula = copula_estimator (fea_clean) ;
            %Sigma_copula = copula_estimator (fea) ;
            %Sigma_copula = Sigma_copula - Sigma_n^2 .* eye(RVn) ;
            
            Sigma_copula = cov(fea) - Sigma_n^2 .* eye(RVn) ; 
            [V,D] = eig(Sigma_copula) ; 
            D(D<0) = 0.01 ;  Sigma_copula_cov = V*D*V' ;  Sigma_copula = corrcov(Sigma_copula_cov);
            MV_RND = mvnrnd(zeros(1,RVn) , Sigma_copula_cov, 1000) ;
            Sigma_copula_hyper_cube = copula_estimator (MV_RND)
            %Sigma_copula = copula_estimator (fea_Lee)
            
%             if(i == length(decomp))
%                 Monte_corr =  cov(fea) - Sigma_n^2 .* eye(RVn)
%                 %Monte_corr = corrcov(Monte_corr) 
%                 Sigma_copula = Monte_corr ; 
%                 Sigma_copula = corrcov(Sigma_copula)
%             end
            %Monte_X = mvnrnd(zeros(1,RVn) , Monte_corr, 1000) ;
            %sz_monte_X = size(Monte_X) ;
            pd_std_n = makedist('Normal', 'mu', 0 , 'sigma', 1); 
            %G_Monte = cdf(pd_std_n,Monte_X) ; 
            %sz_G_monte = size(G_Monte) ;
            %Sigma_copula = G_Monte'*G_Monte/1000 ;
            %Sigma_copula = copulafit('Gaussian', [u,v,k])
            %Sigma_copula = [1]
            %Sigma_copula = eye(RVn) % independent !
            coeff_1 = 1 ; coeff_neigh = 3;
            x= fea(:,coeff_1); y = fea(:,coeff_neigh);
            u = ksdensity(x,x,'function','cdf');
            v = ksdensity(y,y,'function','cdf');
            fig_copula_hyper = figure();
            %scatterhist(u,v)
            scatter(u,v)
            xlabel('u')
            ylabel('v')
            title(['Level: ', num2str(level), 'dir: ', num2str(dir), ' Coeffs: ',num2str(coeff_1) ,',',num2str(coeff_neigh)  ])
            colormap('gray')
            rhohat = copulafit('Gaussian',[u v])
            r = copularnd('Gaussian',rhohat,1000);
            u1 = r(:,1);
            v1 = r(:,2);
            hold on
            %figure;
            %scatterhist(u1,v1)
            scatter(u1,v1)
            legend('Real Data','Gaussian Copula', 'Location', [0.2 0.8 0.15 0.05])
            Filename = join([img_name,string(method_name),"CopulaHyperCube","Scale",num2str(level),"Dir",num2str(dir),"Coeff1",num2str(coeff_1),"Coeff2",num2str(coeff_neigh)],"_") ;
            %Filename = join ([Filename ,string(num2str(size(RandomVariables_lst,2))) ] , "_") ;
            Filename_copula_hyper_png = join([Filename, "png"], ".");
            Filename_copula_hyper_eps = join([Filename, "eps"], ".");
            Filename_copula_hyper_fig = join([Filename, "fig"], ".");
            %imwrite(Image.cdata, join([saving_path, Filename,size(RandomVariables_lst,2), ".png"],"") );
            saveas(fig_copula_hyper,fullfile(convertStringsToChars(saving_path_CopulaHyper),convertStringsToChars(Filename_copula_hyper_png)));
            saveas(fig_copula_hyper,fullfile(convertStringsToChars(saving_path_CopulaHyper),convertStringsToChars(Filename_copula_hyper_eps)));
            saveas(fig_copula_hyper,fullfile(convertStringsToChars(saving_path_CopulaHyper),convertStringsToChars(Filename_copula_hyper_fig)));
            %xlabel('u')
            %ylabel('v')
            %title(['Level: ', num2str(level), 'dir: ', num2str(dir)])
            %colormap('gray')
            %set(get(gca,'children'),'marker','*')
            
%             %% BCGM
%             if (strcmp(method_name, 'BCGM'))
%                 
%                 
%                 
%                 p = stblfit(fea(:,1))
% 
%                 
%                 %% noisy version
%                 [alpha,scale] = SaSGfit_manual(fea(:,1),Sigma_n);
%                 scale = SaSGecfMomFit(fea(:,1),[1:50],alpha,Sigma_n);
%                 scale = abs(scale)
%                  alpha = p(1);
% %                 scale = p(3) ; 
%                 eps = (4-alpha^2) / (3*alpha^2)
%                 %Sigma_copula = corrcov((1-eps)^2 .* Sigma_copula);
%                 %Z = mvnrnd(zeros(1,RVn),Monte_corr,1000);
%                 %corr_coef_Z  = corrcoef(Z)
%                 %U = normcdf(Z);
%                 %U_corr = corrcoef(U)
%                 %rho = copulaparam('Gaussian',corr(MV_RND))
% %                 figure ;
% %                 %p_mle = mle(fea(:,1),'distribution','Stable')
% %                 %pp_stbl = @(data,alpha,beta,gamma,delta)cdf('Stable',data,alpha,beta,gamma,delta);
% %                 %p_mle = mle(fea(:,1),'distribution','tLocationScale');
% %                 %t = @(data,mu,sig,df)cdf('tLocationScale',data,mu,sig,df);
% %                 %h = probplot(gca,pp_stbl,p_mle);
% %                 pd_stbl = makedist('Stable','alpha',alpha,'beta',0, 'gam',p(3) , 'delta',0 );
% %                 title('{\bf Stable QQ plot}')
% %                 qqplot(fea(:,1),pd_stbl );
% %                 figure 
% %                 title('{\bf Normal QQ plot}')
% %                 qqplot(fea(:,1) );
%                 %probplot(fea(:,1));
% %                 h.Color = 'r';
% %                 h.LineStyle = '-';
% %                 title('{\bf Probability Plot}')
% %                 legend('Normal','Data','t','Location','NW')
%                 
%                 
%                 
%                 if (size(Sigma_copula_hyper_cube,1)==1)
%                     Sigma_copula = [1];
%                 else
%                     U = copularnd('gaussian',Sigma_copula_hyper_cube,1000);
%                     %Monte_X = eps.*cauchyinv(U, 0, scale) + (1-eps).* norminv(U,0,scale);
%                     %Monte_X_corr = corrcoef(Monte_X)
%                     G_Monte = icdf(pd_std_n,U) ; 
%                     Sigma_copula = G_Monte'*G_Monte/1000  
%                 end
%                 
%                 JointPrior = Copula_multivariate(BCGM_pdf(eps,scale),Sigma_copula,RVn);
%                
%                 
%                 %% plotting BCGM
% %                 figure()
% %                 fh = fplot(BCGM_pdf(eps,scale));
% %                 fh.LineWidth = 2 ;
% %                 hold on
% %                 [ff,xx]=hist(fea(:,1),50);
% %                 plot(xx,ff/trapz(xx,ff),'*')
% %                 hold off
%             end
%             %% TLS
%             if (strcmp(method_name , 'TLS'))
%                 pd = fitdist(fea_Lee(:,1),'tLocationScale');
%                 mu_tls = pd.mu 
%                 sigma_tls = pd.sigma  
%                 nu_tls = pd.nu 
%                 JointPrior = Copula_multivariate2(tls(pd.mu,pd.sigma,pd.nu),Sigma_copula,RVn) ;
%                 params_tls = paramci(pd) ;
%                 params_tls = params_tls' ;
%                 
%                 line_TLS_statistics = {level,dir,pd.mu,mat2str(params_tls(1,:),5),...
%                     pd.sigma,mat2str(params_tls(2,:),5),...
%                     pd.nu,mat2str(params_tls(3,:),5),0.0,0.0,Sigma_n};
%                 %% plotting TLS
% %                 figure()
% %                 fh = fplot(tls(pd.mu,pd.sigma,pd.nu));
% %                 fh.LineWidth = 3 ;
% %                 hold on
% %                 [ff,xx]=hist(fea_Lee(:,1),1000, 'LineWidth' , 2);
% %                 plot(xx,ff/trapz(xx,ff))
% %                 hold on
% %                 pd_normal = makedist('Normal','mu', 0,  'sigma' , pd.sigma+.2) ;
% %                 pdf_normal = pdf(pd_normal,-1:.01:1);
% %                 plot(-1:.01:1,pdf_normal,'LineWidth',1, 'color','g') ; 
% %                 pd_TS = makedist('tLocationScale','mu', 0,  'sigma' , 1,'nu',pd.nu) ;
% %                 pdf_TS = pdf(pd_TS,-3:.1:3);
% %                 plot(-3:.1:3,pdf_TS,'LineWidth',1)
% %                 legend(['t-loc-scale $\hat{\nu} = $' ,num2str(pd.nu) , '$ \hat{\sigma} = $' , num2str(pd.sigma)],...
% %                     'Emprical PDF' , 'Normal'...
% %                     ,['t student $\hat{\nu} = $' ,num2str(pd.nu)],'interpreter', 'latex');
%             end
%             if (strcmp(method_name , 'NIG'))
%                 %pd = fitdist(fea(:,1),'tLocationScale');
%                 [ alpha,delta ] = nig_fit_moment( fea(:,1),Sigma_n^2 )
%                 JointPrior = Copula_multivariate(NIG_pdf ( alpha+.01, 0, delta+.01, 0),Sigma_copula,RVn);
%                 %% plotting NIG
% %                 figure()
% %                 fh = fplot(NIG_pdf ( alpha, 0, delta, 0));
% %                 fh.LineWidth = 2 ;
% %                 hold on
% %                 [ff,xx]=hist(fea(:,1),50);
% %                 plot(xx,ff/trapz(xx,ff),'*')
% %                 hold off
%             end
%     %         Joint = matlabFunction (JointPrior)
%             %Joint(1,2,3)
%             %cdf = int (matlabFunction( simplify(tls(pd.mu,pd.sigma,pd.nu))),-inf,sym('x1'))
% 
%             %denoised_sacale_level = scale_level_denoising(XX_scale_level_mtb,RVn_lst,JointPrior, Sigma_n,L,gpu_flg);
%             denoised_sacale_level = scale_level_conv_denoising(XX_scale_level_mtb, RVn_lst,JointPrior, Sigma_n,xmin,xmax,integral_bins ,L,gpu_flg);
% 
% 
% 
%             %denoised_sacale_level = scale_level_conv_denoising_edited(XX_scale_level_mtb,dst{level}(:,:,dir), RVn_lst,JointPrior, Sigma_n,xmin,xmax,integral_bins ,L,gpu_flg);
%             %denoised_sacale_level = scale_level_conv_denoising_edited_causal(XX_scale_level_mtb, RVn_lst,JointPrior, Sigma_n,xmin,xmax,integral_bins ,L,gpu_flg);
%             %dst{level}(:,:,dir) = denoised_sacale_level ;
%             my_dst{level}(:,:,dir) = denoised_sacale_level ;
%             noisy_scale_dir= dst{level}(:,:,dir);
%             clean_scale_dir = dst_clean {level}(:,:,dir) ; 
%             [h,p] = ttest(clean_scale_dir(:),noisy_scale_dir(:));
%             %[h,p] = ttest(denoised_sacale_level(:),noisy_scale_dir(:));
%             line_TLS_statistics{end-2} = p; line_TLS_statistics{end-1} = h;
%             TLS_statistics_table = [TLS_statistics_table ; line_TLS_statistics];
%             %figure()
%             %imagesc(denoised_sacale_level)
%         end
%     end
%     splited_path = split(string(SAR_image_path),"\") 
%     img_name_lst = split(splited_path(end),".") 
%     img_name = img_name_lst(1) 
%     iif = @(varargin) varargin{2 * find([varargin{1:2:end}], 1, 'first')}();
%     dependency_check = @(x) iif( x==1, @() 'indep', x>1, 'dep',x==0, 'estimator+Lee' );
%     RV2table = @(x) iif( size(x,2)==1, @() [x nan nan], size(x,2)==2, [x nan], size(x,2)==3,x  );
%     
%     %xr=nsst_rec1(dst,lpfilt)
%     xr=nsst_rec1(my_dst,lpfilt);
%     
%     
%     
%     img_lee_xr = myLee(uint8(exp(xr))) ; 
%     img_lee_xr = uint8(img_lee_xr) ;
%     %imshow(exp(xr))
%     f1 =figure()
% 
%     imshow(uint8(exp(xr)),[])
%     title('uint8 xr reconstructed')
%     %Image = getframe(gcf);
%     Filename = join([img_name,string(method_name),string(dependency_check(size(RandomVariables_lst,2))),"xr"],"_") ;
%     Filename2 = join ([Filename ,string(num2str(size(RandomVariables_lst,2))),strjoin(string(RandomVariables_lst),"") ] , "_") ;
%     Filename = join([Filename2, "png"], ".");
%     %imwrite(Image.cdata, join([saving_path, Filename,size(RandomVariables_lst,2), ".png"],"") );
%     saveas(f1,fullfile(convertStringsToChars(saving_path),convertStringsToChars(Filename)));
% %     f2=figure()
% % 
% %     imshow(uint8(exp(x_noisy)),[])
% %     title ('uint8 x noisy')
% %     
% %     
% %     %Image = getframe(gcf);
%     f2=figure();
%     imshow(uint8(exp(x_noisy)))
%     Filename = join([img_name,string(method_name),string(dependency_check(size(RandomVariables_lst,2))),"xnoisy"],"_") ;
%     Filename = join ([Filename ,string(num2str(size(RandomVariables_lst,2))) ] , "_") ;
%     Filename = join([Filename, "png"], ".");
%     %imwrite(Image.cdata, join([saving_path, Filename,size(RandomVariables_lst,2), ".png"],"") );
%     saveas(f2,fullfile(convertStringsToChars(saving_path),convertStringsToChars(Filename)));
%     
%     
% %     f3=figure()
% % 
% %     imagesc(xr)
% %     title ('log(xr) reconstructed')
% %     
% %     %    Image = getframe(gcf);
% %     Filename = join([img_name,string(method_name),string(dependency_check(size(RandomVariables_lst,2))),"logxr"],"_") ;
% %     Filename = join ([Filename ,string(num2str(size(RandomVariables_lst,2))) ] , "_") ;
% %     Filename = join([Filename, "png"], ".");
% %     %imwrite(Image.cdata, join([saving_path, Filename,size(RandomVariables_lst,2), ".png"],"") );
% %     saveas(f3,fullfile(convertStringsToChars(saving_path),convertStringsToChars(Filename)));
%     
%     
% %     figure()
% % 
% %     imagesc(x_noisy)
% %     title ('x noisy')
% 
% %     f4 =figure()
% %     imshow(uint8(img_clean),[])
% %     title ('image clean')
% %     %Image = getframe(gcf);
% %     Filename = join([img_name,string(method_name),string(dependency_check(size(RandomVariables_lst,2))),"imgclean"],"_") ;
% %     Filename = join ([Filename ,string(num2str(size(RandomVariables_lst,2))) ] , "_") ;
% %     Filename = join([Filename, "png"], ".");
% %     %imwrite(Image.cdata, join([saving_path, Filename,size(RandomVariables_lst,2), ".png"],"") );
% %     saveas(f4,fullfile(convertStringsToChars(saving_path),convertStringsToChars(Filename)));
%     f_lee_xr = figure() ;
%     imshow(img_lee_xr,[]);
%     title('xr Filtered by Lee');
%     saveas(f_lee_xr,fullfile(convertStringsToChars(saving_path),[convertStringsToChars(Filename2),'xrLee.png']));
% 
%     
%     
%     
%     RVn_lst;
% 
%     MSE_clean_denoisaed = immse(uint8(exp(xr)),  uint8(img_clean)) 
%     MSE_noisy_denoised = immse(uint8(exp(xr)),  uint8(exp(x_noisy))) 
%      
%     [xr_exp_ST,dst_ST,dst_new_ST] = soft_treshold(x_exp,Sigma_n) ;
%     [xr_exp_HT,dst_HT,dst_new_HT] = hard_treshold(x_exp,Sigma_n) ; 
%     
%     
%     f_ST = figure
%     imshow(uint8(xr_exp_ST),[]);
%     title('Soft Treshold');
%     saveas(f_ST,fullfile(convertStringsToChars(saving_path),[convertStringsToChars(Filename2),'xrST.png']));
%     
%     f_HT = figure
%     imshow(uint8(xr_exp_HT),[]);
%     title('Hard Treshold');
%     saveas(f_HT,fullfile(convertStringsToChars(saving_path),[convertStringsToChars(Filename2),'xrHT.png']));
%     
%     
%     line = {splited_path(end),MSE_clean_denoisaed,...
%         psnr(uint8(exp(xr)),uint8(img_clean)),...
%         ssim(uint8(exp(xr)),uint8(img_clean)),...
%         epi((exp(xr)),(img_clean)),ENL(exp(xr)),...
%         method_name,Sigma_noise,...
%         [RV2table(RandomVariables_lst)] ,...
%         dependency_check(size(RandomVariables_lst,2)) } ;
%     
%     line_lee_xr = {splited_path(end),immse(uint8(img_lee_xr),  uint8(img_clean)),...
%         psnr(uint8(img_lee_xr),uint8(img_clean)),...
%         ssim(uint8(img_lee_xr),uint8(img_clean)),...
%         epi(double(img_lee_xr),(img_clean)),ENL(img_lee_xr),...
%         method_name,Sigma_noise,...
%         [RV2table(RandomVariables_lst)] ,...
%         dependency_check(0) } 
%     RV2table(RandomVariables_lst)
%     %NoiseInfo_table = [NoiseInfo_table;line] ;
%     line_ST = {splited_path(end),immse(uint8(xr_exp_ST),  uint8(img_clean)),...
%         psnr(uint8(xr_exp_ST),uint8(img_clean)),...
%         ssim(uint8(xr_exp_ST),uint8(img_clean)),...
%         epi(xr_exp_ST,(img_clean)),ENL(xr_exp_ST),...
%         'SoftT',Sigma_noise,...
%         [RV2table(RandomVariables_lst)] ,...
%         dependency_check(size(RandomVariables_lst,2)) } ;    
%     
%     line_HT = {splited_path(end),immse(uint8(xr_exp_HT),  uint8(img_clean)),...
%         psnr(uint8(xr_exp_HT),uint8(img_clean)),...
%         ssim(uint8(xr_exp_HT),uint8(img_clean)),...
%         epi(xr_exp_HT,(img_clean)),ENL(xr_exp_HT),...
%         'HardT',Sigma_noise,...
%         [RV2table(RandomVariables_lst)] ,...
%         dependency_check(size(RandomVariables_lst,2)) } ;      
%     
%     Yuan_img = Yuan(x_exp,Sigma_noise) ;
%     
%     f_Yuan = figure
%     imshow(uint8(Yuan_img),[]);
%     title('Yuan');
%     saveas(f_Yuan,fullfile(convertStringsToChars(saving_path),[convertStringsToChars(Filename2),'xrYuan.png']));
%     %saveas(f_Yuan,fullfile(convertStringsToChars(saving_path),[convertStringsToChars(Filename2),'xrYuan.png']));
%     
%     line_Yuan = {splited_path(end),immse(uint8(Yuan_img),  uint8(img_clean)),...
%         psnr(uint8(Yuan_img),uint8(img_clean)),...
%         ssim(uint8(Yuan_img),uint8(img_clean)),...
%         epi(Yuan_img,(img_clean)),ENL(Yuan_img),...
%         'Yuan',Sigma_noise,...
%         [RV2table(RandomVariables_lst)] ,...
%         dependency_check(size(RandomVariables_lst,2)) } ;  
%     
%    Liu_img = cs_nsst(x_exp);
%     Liu_img = uint8(Liu_img)  ;
%     Liu_img = imhistmatchn(Liu_img, uint8(Yuan_img))
%     f_Liu = figure
%     imshow(uint8(Liu_img),[]);
%     title('CsNsst');
%     saveas(f_Liu,fullfile(convertStringsToChars(saving_path),[convertStringsToChars(Filename2),'xrLiu.png']));
%     %saveas(f_Yuan,fullfile(convertStringsToChars(saving_path),[convertStringsToChars(Filename2),'xrYuan.png']));
%     
%     line_Liu = {splited_path(end),immse((Liu_img),  uint8(img_clean)),...
%         psnr((Liu_img),uint8(img_clean)),...
%         ssim((Liu_img),uint8(img_clean)),...
%         epi(double(Liu_img),(img_clean)),ENL(double(Liu_img)),...
%         'CsNsst',Sigma_noise,...
%         [RV2table(RandomVariables_lst)] ,...
%         dependency_check(size(RandomVariables_lst,2)) } ;    
%     
%     
%     %plot_io_shearlet_coeffs(dst ,my_dst ,'vie1','C:\Users\Arian\GitBackupThesis\thesis\sample\ThesisCode\Results\io')
%     
%     noise_info = NoiseInfo_table ;
%     res_table=  [NoiseInfo_table;line;line_lee_xr ; line_HT ; line_ST;line_Yuan;line_Liu] ;
%     Info_line = [line;line_lee_xr] ;
%     TLS_table = TLS_statistics_table ; 
%     x_rec = exp(xr - Sigma_noise^2/2) ; 
        end
    end
end