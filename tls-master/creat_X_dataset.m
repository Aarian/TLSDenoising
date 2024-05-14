function [sample_scale] = creat_X_dataset(coefficients,level, direction,neighbors)
% this functions takes shearlet coeffs as input and computes the output as
% X dataset
% input: 
% coefficients -cell array contains all the shearlet coeffs
% level -the level of shearlet array (j)
% direction -direction in each level (0<direction<2^j-1 )
% neighbors -number od neighbors for a specific sample
% output: 
% sample_scale -dataset X + 2 coulmns containing indices of shearlet coeff
    direction
    level
    coefficient = coefficients{level}(:,:,direction);

    coeff_sz = size(coefficient);
    X = ones(neighbors+1+2 , (coeff_sz(1))*(coeff_sz(2)) );
    for i = 1:coeff_sz(1)
        for j = 1:coeff_sz(2)
            if(j == coeff_sz(2) & i ~= coeff_sz(1) )
                X(:,(i-1)*coeff_sz(1) +  j) = [coefficient(i,j)  , 0 , coefficient(i+1,j), 0,i,j]'; 
            elseif i == coeff_sz(1) & j ~= coeff_sz(2) 
                X(:,(i-1)*coeff_sz(1) +  j) = [coefficient(i,j)  , 0 , 0, coefficient(i,j+1),i,j]';
            elseif i == coeff_sz(1) & j == coeff_sz(2)
                X(:,(i-1)*coeff_sz(1) +  j) = [coefficient(i,j)  , 0 , 0, 0,i,j]';
            else
                X(:,(i-1)*coeff_sz(1) +  j) = [coefficient(i,j)  , coefficient(i+1,j+1) , coefficient(i+1,j), coefficient(i,j+1),i,j]';
            end
        end
        
    end
    sample_scale = X;
    
end