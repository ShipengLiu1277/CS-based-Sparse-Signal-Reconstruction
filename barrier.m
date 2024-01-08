function [opt_x,opt_value,inner_it]=barrier(F,X,lambda,p,t,x_hat,error_tol,mu)

    delta = t;
    global obj_val;
    global obj_it;
    global x_tmp;
    
    cnt = 1;
    inner_it = [];
    while (1)
        % 1.Centering step: compute x'(t) by minimizing tf+φ, subject to Ax=b.
        [newx,newvalue,inner_iteration] = backtracking_newton(F,X,lambda,p,delta,x_hat,error_tol);
        x_tmp = [x_tmp,newx(1:p)];
        % 2.Update.x:=x'(t).
        x_hat = newx;  
        opt_x = x_hat; %disp
        opt_value = newvalue; %disp   
        obj_val = [obj_val,newvalue];
        cnt = cnt + 1;
        obj_it = [obj_it, cnt];
        inner_it = [inner_it,inner_iteration];

        disp('t:');
        disp(delta);
        disp('Optimal x(t):');
        disp(opt_x);
        disp('Optimal p(t):');
        disp(opt_value);

        % 3.Stopping criterion. quit if m/t < error_tol.
        if (2*p/delta<error_tol)
            break
        end
        
        % 4.Increase t.t:=μt
        delta = mu*delta; %disp
    end
    
end