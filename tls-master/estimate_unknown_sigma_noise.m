function [Nsig] = estimate_unknown_sigma_noise(img)
    % estimates additive noise sigma
    

x = log(img);

J = 5;
I=sqrt(-1);
load nor_dualtree
% symmetric extension
L = length(x); % length of the original image.
N = L+2^J;     % length after extension.
x = symextend(x,2^(J-1));

[Faf, Fsf] = AntonB;
[af, sf] = dualfilt1;
W = cplxdual2D(x, J, Faf, af);
W = normcoef(W,J,nor);

% Noise variance estimation using robust median estimator..
tmp = W{1}{1}{1}{3};
Nsig = median(abs(tmp(:)))/0.6745


end