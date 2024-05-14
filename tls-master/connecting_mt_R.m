function symbolic_prior = connecting_mt_R (nig_param,Sigma,RVn)


 alpha= nig_param(1)
 beta= nig_param(2)
 mu= nig_param(3)
 delta= nig_param(4)
% mu = 0.034 ; 
% 
% 
% delta = 2.12; 
% 
% alpha = 0.05;
% beta = -0.001; 
% 
% mu = 0 ; 
% 
% 
% delta = 1; 
% 
% alpha = 1;
% beta = 0; 


%prob = NIG_integral (0, inf, alpha, beta, delta, mu);
%prob = int(NIG_pdf(x,alpha,beta,delta,mu),0,inf);
%double(prob)

Sigma =  eye(RVn) ;

joint_fx = Copula_multivariate(NIG_pdf(alpha,beta,delta,mu), Sigma, RVn);
class(joint_fx);
joint_pdf_function_handle= matlabFunction(joint_fx);
%subs(joint_fx, [sym('x1'),sym('x2')], [sym('x'),sym('y')])
%integral2 ( joint_pdf_function_handle,0,1 , -inf,inf);

%integral3 ( joint_pdf_function_handle,-inf,inf , 0,1, -inf,inf)

symbolic_prior = joint_fx;

% Sigma = eye(3) ;
% joint_fx = Copula_multivariate(NIG_pdf(alpha,beta,delta,mu), Sigma, 3)
% integral3 ( @(x1,x2,x3)joint_fx(x1,x2,x3) ,-inf,inf , -inf,inf, -inf, inf)
end
