function []=show_shearlet_coeffs(coeffs_img,Title)
        sub_plot_x=2;
        sub_plot_y=1;
        coeff_sz = size(coeffs_img);
    for coeff_idx = 1:coeff_sz(3) 
        subplot(sub_plot_x, sub_plot_y,coeff_idx);
        %imshow(coeffs_img(:,:,coeff_idx));
        imagesc(coeffs_img(:,:,coeff_idx));
        title([' Coeff Idx= ',num2str(coeff_idx)])
        sgtitle(Title)
    end

end