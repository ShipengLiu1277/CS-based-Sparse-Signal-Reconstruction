clc
clear
close all
load('cs.mat')

F_us0 = F_us(sampling_mask==1,:);
X_us0 = X_us(sampling_mask==1);
F_us0_real = real(F_us0);
F_us0_imag = imag(F_us0);
X_us0_real = real(X_us0);
X_us0_imag = imag(X_us0);

Aeq = [F_us0_real;F_us0_imag];
Beq = [X_us0_real;X_us0_imag];

x_hat0 = Aeq\Beq;
% x_hat0 = (Aeq'*Aeq)\Aeq'*Beq;
x_hat0_plus = abs(x_hat0);
x_hat0_plus(x_hat0<=0) = 0;
x_hat0_minus = abs(x_hat0);
x_hat0_minus(x_hat0>=0) = 0;
z = [x_hat0_plus;x_hat0_minus];
[numRows_z,numCols_z]=size(z);

lambda0 = 0.001;
c = lambda0*ones(numRows_z,1) +[-Aeq'*Beq;Aeq'*Beq];
B = [Aeq'*Aeq,-Aeq'*Aeq;-Aeq'*Aeq,Aeq'*Aeq];

% % using "quadprog" to solve the quadratic programming problem
% zx = quadprog(B,c,eye(numRows_z)*(-1),zeros(numRows_z,1))
% stem((zx(1:128)-zx(129:256)))
% hold on
% stem(x)
% hold off

iter = 1000;
alpha0 = 0.4;
beta0 = 0.8;
mem1 = zeros(iter+1,numRows_z);
mem1(1,:) = z;
mem0 = zeros(iter+1,numRows_z/2);
mem0(1,:) = z(1:128)-z(129:256);
t_start = cputime;
for t=1:iter
    stepsize = 5;
    while (c'*(z+stepsize*(-(c+B*z)))+0.5*(z+stepsize*(-(c+B*z)))'*B*(z+stepsize*(-(c+B*z))))>((c'*z+0.5*z'*B*z)+(alpha0*stepsize*(c+B*z)'*(-(c+B*z))))
        stepsize = stepsize*beta0;
    end
    z_update = stepsize*(c+B*z);
    if z_update < 1e-7
        break;
    end
    z = z - z_update;
    z = max(z,0);
    mem1(t+1,:) = z;
    mem0(t+1,:) = z(1:128)-z(129:256);
end
t_end = cputime;
time0 = t_end - t_start;


error5 = cal_error(mem0,x);


x_hat0 = z(1:128)-z(129:256);



stem(x_hat0)
hold on
stem(x)
hold off

plot(0:iter,error5)
ylabel('MSE')
xlabel('Iteration')


f0 = cal_f0_quad(mem1,c,B);
plot(0:iter,f0)
ylabel('f_0')
xlabel('Iteration')

time0;

function output = cal_error(mem,x_star)
    [numRows,~] = size(mem);
    temp = zeros(numRows,1);
    for k = 1:numRows
        temp(k) = mean((mem(k,:)'-x_star).^2);
        % temp(k) = sum((mem(k,:)'-x_star).^2)/128;
    end
    output = temp;
end

function output = cal_f0_quad(mem,c0,B0)
    [numRows,~] = size(mem);
    temp = zeros(numRows,1);
    for k = 1:numRows
        z_temp = mem(k,:)';
        temp(k) = c0'*z_temp + 0.5*z_temp'*B0*z_temp;
    end
    output = temp;
end