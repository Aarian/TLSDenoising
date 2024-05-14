function [upper,downer] = MMSE(Prior_sym, RVn,sigma_n, idx_coeff)
    %% Parameters : 
    % Prior_sym   : symbolic version of Prior
    % RVn         : number of RVs or the neighbors (i.e. d)
    % Sigma_noise : vaiance of noise   
    
    %% return     :
        %Symbolic version of MMSE (upper/downer)
    
    %%  
    X = sym('x',[RVn 1]);
    Y = sym('y',[RVn 1]);
    %syms sigma_n
    up_expr = X(idx_coeff)* (2*pi*sigma_n^2)^(-RVn/2) * exp(-(Y-X)'* 1/sigma_n^2 * eye(RVn) * (Y-X)) * Prior_sym ;
    %up_expr_functionHandle = matlabFunction(up_expr) ; 
    %size(up_expr)
    down_expr = (2*pi*sigma_n^2)^(-RVn/2) * exp(-(Y-X)'* 1/sigma_n^2 * eye(RVn) * (Y-X)) * Prior_sym ;
    %down_expr_functionHandle = matlabFunction(down_expr);
%     if (RVn==2)
%         MMSE_sym = integral2 (@(x1,x2) up_expr_functionHandle,-inf,inf , -inf,inf) / integral2 (@(x1,x2) down_expr_functionHandle,-inf,inf , -inf,inf) ;
%         
%     end
%     if(RVn==3)
%          MMSE_sym = integral3 ( @(x1,x2,x3) up_expr_functionHandle,-inf,inf , -inf,inf, -inf,inf) / integral3 (@(x1,x2,x3) down_expr_functionHandle, -inf,inf,-inf,inf , -inf,inf) ;
%     end
    
    upper  = up_expr ;
    downer = down_expr ;
end