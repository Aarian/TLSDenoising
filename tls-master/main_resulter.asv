
RV_cells = {[1]};
NL = [2]
method_cells = {'TLS'};
img_paths =["images_SAR_real3.jpg"]
tabl = table({''},[.0],[.0],[.0],[.0],[.0],{''},[.2],[nan nan nan],{'Unkonwn'},'VariableNames',{'ImgName','MSE','PSNR','SSIM', 'EPI','ENL', 'PriorDist','Sigma_noise','RV_lst', 'dependency'} )
for mgg = 1:size(img_paths,2)
for i_nl  = 1:size(NL,2) 
    Sig_n = sqrt(1/NL(i_nl));
    %Sig_n = .45
    %Sig_n = 0 % when the original version is not available
    for i = 1:size(RV_cells,2)
        RV_i = RV_cells{i}
        for j = 1:size(method_cells,2)
            method_j =  method_cells{j}
            if (size(RV_cells{i},2)==1)
                [noise_info,res_table,Info_line,x_rec,dst,my_dst,dst_clean,TLS_stat_table] = resulter (char(img_paths(mgg)), Sig_n,RV_cells{i}, [1 2 2], method_cells{j})
                tabl = [tabl ; res_table] 
            else
                if (strcmp( method_cells{j},'TLS')==0)
                    [noise_info,res_table,Info_line,x_rec,dst,my_dst,dst_clean,~]= resulter (char(img_paths(mgg)), Sig_n,RV_cells{i}, [1 2 2], method_cells{j});
                    tabl = [tabl ; Info_line]  ;
                end
            end
        end
    end
end
end