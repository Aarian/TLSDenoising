function estimated_X = integral_MMSE(upper, downer,Y_obvs,RVn,gpu_flag)
    %% Parameters
        % upper : symbolic expr upper
        % dowmer : symbolic expr downer
        % Y_obvs : observations of coeffs columns vector
        % RVn : number of RVs
        % gpu_flag : copute the MMSE on gpu
   %% retuen
    % compute integral!
    %% 
    if (size(Y_obvs,1) ~= RVn)
          error('size obvs do not match the RVn');
    end 
    if gpu_flag == 1
        xmin = -1 ;
        xmax = 1 ; 
        integral_bins = 20 ; 
        if (RVn ==2)
            syms y1 y2;
            y1 = Y_obvs(1);
            y2 = Y_obvs(2);
            subs(upper);
            %feval(symengine, upper, y1,y2)
            hdl = matlabFunction(subs(upper));
            %passed = @(x1,x2)hdl(x1,x2,Y_obvs(1),Y_obvs(2))
            upp =  new_integral (hdl, xmin, xmax,integral_bins, 2,RVn);
            downn =  new_integral (matlabFunction(subs(downer)), xmin, xmax,integral_bins, 2,RVn);
            estimated_X = upp/downn ; 
        end
        if (RVn ==3)
            syms y1 y2 y3
            y1 = Y_obvs(1);
            y2 = Y_obvs(2);
            y3 = Y_obvs(3);
            upp =  gpu_integral (subs(upper), xmin, xmax,integral_bins, 3,RVn);
            downn =  gpu_integral (subs(downer), xmin, xmax,integral_bins, 3,RVn);
            estimated_X = upp/downn ; 
        end
    else
    
         %up_expr_functionHandle = matlabFunction(upper) ; 
         %down_expr_functionHandle = matlabFunction(downer);
         xmin = -inf ;
         xmax = inf ; 
         if(RVn ==2)
            syms y1 y2;
            y1 = Y_obvs(1);
            y2 = Y_obvs(2);
            up_expr_functionHandle = matlabFunction(subs(upper)) ;
            down_expr_functionHandle = matlabFunction(subs(downer));
            upper_int = integral(@(x2)integral(@(x1)up_expr_functionHandle(x1,x2),xmin,xmax,'ArrayValued',true),xmin,xmax,'ArrayValued',true);
            %upper_int =integral2 (up_expr_functionHandle,x_min,x_max , x_min,x_max);
            downer_int = integral(@(x2)integral(@(x1)down_expr_functionHandle(x1,x2),xmin,xmax,'ArrayValued',true),xmin,xmax,'ArrayValued',true);
            %downer_int =integral2 ( down_expr_functionHandle,x_min,x_max , x_min,x_max);
            MMSE_div = upper_int  / downer_int ;
            estimated_X = MMSE_div;
         end

         if(RVn ==3)
            syms y1 y2 y3;
            y1 = Y_obvs(1);
            y2 = Y_obvs(2);
            y3 = Y_obvs(3);
            up_expr_functionHandle = matlabFunction(subs(upper)) ;
            down_expr_functionHandle = matlabFunction(subs(downer));
            %MMSE_div = up_expr_functionHandle  / down_expr_functionHandle ;
            upper_int = integral(@(x3)integral(@(x2)integral(@(x1)up_expr_functionHandle(x1,x2,x3),xmin,xmax,'ArrayValued',true),xmin,xmax,'ArrayValued',true),xmin,xmax,'ArrayValued',true)
            downer_int = integral(@(x3)integral(@(x2)integral(@(x1)down_expr_functionHandle(x1,x2,x3),xmin,xmax,'ArrayValued',true),xmin,xmax,'ArrayValued',true),xmin,xmax,'ArrayValued',true)
            %upper_int =integral3 (up_expr_functionHandle,x_min,x_max , x_min,x_max,x_min,x_max);
            %downer_int =integral3 ( down_expr_functionHandle,x_min,x_max ,x_min,x_max,x_min,x_max);
            MMSE_div = upper_int/downer_int ;
            estimated_X = MMSE_div ;
         end
    end
end