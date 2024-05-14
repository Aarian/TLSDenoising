function joint_fx = Copula_multivariate2(MargDist, Sigma_cop, RVn)
    % MargDist : marginal distribution of coeff (based on data set cinstruction all the Margins are the same)
    % Sigma_cop : covariance matrix copula 
    % RVn : number of RVs
    
    % joint_fx : symbolic joint_fx (need to be converted to function handle)
    %sym x1 x2
    if size(Sigma_cop ,1) ~= RVn
        error('Error. Miss matching in Dim Copula and Size RVs')
    end
    
    
    X = sym('x',[RVn 1]);
    IPHI = sym('iphi',[RVn 1]) ;
    U = sym('u',[RVn 1]) 
    if (RVn == 1)
        Marge1 = subs(MargDist, [sym('x')], [sym('x1')]);
        joint_fx = Marge1;
    end
    if(RVn==2) 
       Marge1 = subs(MargDist, [sym('x')], [sym('x1')]);
       Marge2 = subs(MargDist, [sym('x')], [sym('x2')]);
       %dett=-X' * (inv(Sigma_cop)-eye(RVn)) *X/2 
       joint_fx_temp  = 1/(det(Sigma_cop)^.5) *exp(-IPHI' * (inv(Sigma_cop)-eye(RVn)) *IPHI/2 ) * Marge1 * Marge2 ;
       expr_sym_iphi1 = -2^.5 * erfcinv(2*sym('u1'));
       expr_sym_iphi2 = -2^.5 * erfcinv(2*sym('u2'));
       joint_fx_temp = subs(joint_fx_temp, [sym('iphi1') , sym('iphi2')], [expr_sym_iphi1,expr_sym_iphi2]);
        %cdfnig_1 = int (matlabFunction(MargDist),-inf,sym('x1'));
        cdfnig_1 = (int(Marge1))
        %cdfnig_2 = int (matlabFunction(MargDist),-inf,sym('x2'));
        cdfnig_2 =  (int(Marge2));
       joint_fx = subs(joint_fx_temp, [sym('u1') , sym('u2')], [cdfnig_1,cdfnig_2]);
       string(joint_fx);
        %joint_fx = subs(joint_fx1, [sym('iphi1') , sym('iphi2')], [expr_sym_iphi1,expr_sym_iphi2])
    end
    if(RVn==3) 
        disp('im here')
       Marge1 = subs(MargDist, [sym('x')], [sym('x1')]);
       Marge2 = subs(MargDist, [sym('x')], [sym('x2')]);
       Marge3 = subs(MargDist, [sym('x')], [sym('x3')]);
       % joint_fx  = 1/(det(Sigma_cop)^.5) *exp(-X' * (inv(Sigma_cop)-eye(RVn)) *X/2 ) * Marge1 * Marge2* Marge3
        joint_fx_temp  = 1/(det(Sigma_cop)^.5) *exp(-IPHI' * (inv(Sigma_cop)-eye(RVn)) *IPHI/2 ) * Marge1 * Marge2 * Marge3;
       expr_sym_iphi1 = -2^.5 * erfcinv(2*sym('u1'));
       expr_sym_iphi2 = -2^.5 * erfcinv(2*sym('u2'));
       expr_sym_iphi3 = -2^.5 * erfcinv(2*sym('u3'));
       joint_fx_temp = subs(joint_fx_temp, [sym('iphi1') , sym('iphi2'),sym('iphi3')], [expr_sym_iphi1,expr_sym_iphi2,expr_sym_iphi3]);
        %cdfnig_1 = int (matlabFunction(MargDist),-inf,sym('x1'));
        %cdfnig_2 = int (matlabFunction(MargDist),-inf,sym('x2'));
        %cdfnig_3 = int (matlabFunction(MargDist),-inf,sym('x3'));
        cdfnig_1 = (int(Marge1));
        cdfnig_2 = (int(Marge2));
        cdfnig_3 = (int(Marge3));
       joint_fx = subs(joint_fx_temp, [sym('u1') , sym('u2'), sym('u3')], [cdfnig_1,cdfnig_2,cdfnig_3]);
    end
    
    %gpuDevice
    
    %Marge1 = subs(MargDist, [sym('x')], [sym('x2')])
    %joint_fx  = 1/(det(Sigma_cop)^.5) *exp(-X' .* (inv(Sigma_cop)-eye(RVn)) .*X/2 ) * subs(MargDist1, [sym('x')], [sym('x1')]) *subs(MargDist2, [sym('x')], [sym('x2')]) ;
end