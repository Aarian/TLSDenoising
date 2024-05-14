function Z  = gpu_integral (symbolic_function, xmin, xmax,integral_bins, integ_num,RVn)
    %% parameters : 
        % symbolic_function : integrated function 
        % x min :  lower bound 
        % x max : hogher bound 
        % integral_bin : broken number for evaluating function (integer)
        % RVn : number of RVs (number of integrals)
    function_handle= matlabFunction(symbolic_function);
    if (integ_num ~= RVn)
        error ('plz check the number of integrals with RVn!!(they musy be equal)')
    end
    if (RVn ~=2 & RVn ~=3)
       error ('this computer only computes double or triple integrals :)') 
    end
    if (RVn == 2)
        
       %disp('im in gpu:)') 
       x1_space = gpuArray.linspace(xmin,xmax,integral_bins)  ;
       x2_space = gpuArray.linspace(xmin,xmax,integral_bins) ; 
       xspacing = (xmax-xmin)/integral_bins ;
       %P_test= Pcode_test(function_handle,xmin,xmax,integral_bins)
       [X Y] = meshgrid(x1_space,x2_space) ; 
       F = arrayfun(function_handle,X, Y)  ;
       
       %save('C:\Users\Arian\sample\ThesisCode\SharedAreaMtbPy\F.mat','F','xspacing')
       %Z = py.Mylib.fuu(reshape(F.',1,[]));
       F_gpu = gpuArray(F);
       %check_gpu = existsOnGPU(F_gpu);
       Z1 = trapz(F_gpu) * xspacing  ; 
       Z = gather(trapz(Z1) * xspacing)  ;
    end
    
    if (RVn == 3)
        
       x1_space = linspace(xmin,xmax,integral_bins) ; 
       x2_space = linspace(xmin,xmax,integral_bins) ; 
       x3_space = linspace(xmin,xmax,integral_bins) ; 
       xspacing = (xmax-xmin)/integral_bins ;
       [X Y Z] = meshgrid(x1_space,x2_space,x3_space) ; 
       F = arrayfun(function_handle,X, Y,Z)  ;
       F_gpu = gpuArray(F);
       Z1 = trapz(F_gpu) * xspacing  ;
       Z2 = trapz(Z1) * xspacing ; 
       Z = gather(trapz(Z2) * xspacing)  ;
    end

end