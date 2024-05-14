function [c_level_dir] = Coeff_Level_Direction(coefficients,level, direction)
% returns shear let coeff on  desired -level direction
% X dataset
% input: 
% coefficients -cell array contains all the shearlet coeffs
% level -the level of shearlet array (j)
% direction -direction in each level (0<direction<2^j-1 )
% 
% output: 
% sample_scale -dataset X + 2 coulmns containing indices of shearlet coeff
    c_level_dir = coefficients{level}(:,:,direction);
end