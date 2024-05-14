function copula_sigma_estimate = copula_estimator (column_features)
    sz2 = size(column_features,2); 
    tmp = [] ;
    if (sz2 == 1)
        copula_sigma_estimate = [1];
    
    else
        for j = 1:sz2
           u = ksdensity(column_features(:,j),column_features(:,j),'function', 'cdf');
           tmp = [u,tmp] ;
        end
    copula_sigma_estimate = copulafit('Gaussian', tmp);    
    end
    
end