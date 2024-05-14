function P_noise_symbolic = P_noise( sigma_n, RVn)
    % 
    % Sigma_n : std of noise 
    % RVn : number of RVs
    
    % P_noise_symbolic : symbolic joint_fx (need to be converted to function handle)
    %sym x1 x2

    
    
    X = sym('x',[RVn 1]);
    P_noise_symbolic = (2*pi*sigma_n^2)^(-RVn/2) * exp(-(X)'* 1/sigma_n^2 * eye(RVn) * (X));
end