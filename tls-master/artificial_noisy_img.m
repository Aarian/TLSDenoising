function img_noisy = artificial_noisy_img (img_path, Sigma_n,method )
    x=double(imread(img_path));

    [L L]=size(x);

    % Create noisy image
    if(method == "logTr")
        disp('im in logTR')
        img_noisy = log(x+1) + Sigma_n.*randn(L,L);
    else
        img_noisy = x + Sigma_n.*randn(L,L);
	end
end