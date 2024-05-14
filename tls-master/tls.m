function pdf = tls (mu, sigma,nu) 
    syms x 
   pdf = ( gamma(.5*(nu+1)) /(sigma * sqrt(nu*pi)*gamma(.5*nu)) ) * ( (1+ (1/nu)*((x-mu)/sigma)^2)^(-.5*(nu+1)) )
end