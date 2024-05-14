function []= hist_dst_coeffs(dst,img_tlt)
    sz_cell = size(dst,2); 
    %mkdir(['C:\Users\Arian\sample\Resualts\dep\rep_img\', img_tlt]) ;
    
    for i = 1:sz_cell
        
        if (i==1)
            figure()
            
            hist(dst{1},100);
            title(['Approximation level  ', img_tlt])
            %saveas(gcf,['C:\Users\Arian\sample\Resualts\dep\rep_img\', img_tlt,'img',num2str(i)],'png')
        else
            figure()
            coeff3d = size(dst{i})
            for j = 1:coeff3d(3)
                
                subplot(4, 4,j);
                
                hist(dst{i}(:,:,j),100);
                title(['scale,level = ', num2str([i, j])] )
               
            end 
             %saveas(gcf,['C:\Users\Arian\sample\Resualts\dep\rep_img\', img_tlt,'\img',num2str(i),num2str(j)],'png')
        end



end