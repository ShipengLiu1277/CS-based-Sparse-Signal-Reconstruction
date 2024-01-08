clc
clear
close all
load('cs.mat')

% F_us, sampling mask, X_us are known
F_us0 = F_us(sampling_mask==1,:);
X_us0 = X_us(sampling_mask==1);
F_us0_real = real(F_us0);
F_us0_imag = imag(F_us0);
X_us0_real = real(X_us0);
X_us0_imag = imag(X_us0);

% Problem: min |x| s.t. F_us_0*x = X_us_0

% initialization
Aeq = [F_us0_real;F_us0_imag];
Beq = [X_us0_real;X_us0_imag];

[numRows_x,~] = size(x);
[~,numCols_F_us0] = size(F_us0);

I = eye(numCols_F_us0);
Orth_mat = (I-Aeq'/(Aeq*Aeq')*Aeq);
iter = 1000; % number of iterations
tol=1e-4;

alpha_k0 = 0.01;

mem0 = zeros(iter+1,numRows_x);
% x_hat0 = Aeq\Beq;
x_hat0 = (Aeq'*Aeq)\Aeq'*Beq;
mem0(1,:) = x_hat0;

t_start = cputime;
for t=1:iter
    update_x_hat = alpha_k0*Orth_mat*sign(x_hat0);
    if abs(update_x_hat) < tol
        break;
    end
    x_hat0 = x_hat0 - update_x_hat;
    mem0(t+1,:) = x_hat0;
end

t_end = cputime;
time0 = t_end - t_start;

error0 = cal_error(mem0,x);

stem(x_hat0)
hold on
stem(x)
legend('x\_{hat} ( {\alpha}_k = 0.01)','x')
hold off

alpha_k1 = 0.001;

mem1 = zeros(iter+1,numRows_x);
% x_hat1 = Aeq\Beq;
x_hat1 = (Aeq'*Aeq)\Aeq'*Beq;
mem1(1,:) = x_hat1;

t_start = cputime;
for t=1:iter
    update_x_hat = alpha_k1*Orth_mat*sign(x_hat1);
    if abs(update_x_hat) < tol
        break;
    end
    x_hat1 = x_hat1 - update_x_hat;
    mem1(t+1,:) = x_hat1;
end

t_end = cputime;
time1 = t_end - t_start;

error1 = cal_error(mem1,x);

stem(x_hat1)
hold on
stem(x)
legend('x\_{hat} ( {\alpha}_k = 0.001)','x')
hold off

alpha_k2 = 0.1;

mem2 = zeros(iter+1,numRows_x);
% x_hat2 = Aeq\Beq;
x_hat2 = (Aeq'*Aeq)\Aeq'*Beq;

mem2(1,:) = x_hat2;

t_start = cputime;
for t=1:iter
    update_x_hat = alpha_k2*Orth_mat*sign(x_hat2);
    if abs(update_x_hat) < tol
        break;
    end
    x_hat2 = x_hat2 - update_x_hat;
    mem2(t+1,:) = x_hat2;
end

t_end = cputime;
time2 = t_end - t_start;

error2 = cal_error(mem2,x);

stem(x_hat2)
hold on
stem(x)
legend('x\_{hat} ( {\alpha}_k = 0.1)','x')
hold off

% alpha_k3 = 1/k
mem3 = zeros(iter+1,numRows_x);
% x_hat3 = Aeq\Beq;
x_hat3 = (Aeq'*Aeq)\Aeq'*Beq;

mem3(1,:) = x_hat3;

t_start = cputime;
for t=1:iter
    alpha_k3 = 1/t;
    update_x_hat = alpha_k3*Orth_mat*sign(x_hat3);
    if abs(update_x_hat) < tol
        break;
    end
    x_hat3 = x_hat3 - update_x_hat;
    mem3(t+1,:) = x_hat3;
end

t_end = cputime;
time3 = t_end - t_start;

error3 = cal_error(mem3,x);

stem(x_hat3)
hold on
stem(x)
legend('x\_{hat} ( {\alpha}_k = 1/k)','x')
hold off

% alpha_k4 = 0.1/(k)^0.5
mem4 = zeros(iter+1,numRows_x);
% x_hat4 = Aeq\Beq;
x_hat4 = (Aeq'*Aeq)\Aeq'*Beq;

mem4(1,:) = x_hat4;

t_start = cputime;
for t=1:iter
    alpha_k4 = 0.1*(t)^(-0.5);
    update_x_hat = alpha_k4*Orth_mat*sign(x_hat4);
    if abs(update_x_hat) < tol
        break;
    end
    x_hat4 = x_hat4 - update_x_hat;
    mem4(t+1,:) = x_hat4;
end

t_end = cputime;
time4 = t_end - t_start;

error4 = cal_error(mem4,x);

stem(x_hat4)
hold on
stem(x)
legend('x\_{hat} ( {\alpha}_k = 1/k^{0.5})','x')
hold off

plot(0:iter,error0)
hold on
plot(0:iter,error1)
plot(0:iter,error2)
plot(0:iter,error3)
plot(0:iter,error4)
ylabel('MSE')
xlabel('Iteration')
legend('{\alpha}_k = 0.01','{\alpha}_k = 0.001', ...
    '{\alpha}_k = 0.1','{\alpha}_k = 1/k','{\alpha}_k = 1/k^{0.5}')
% xlim([0 100])
hold off

time0;
time1;
time2;
time3;
time4;

avg_time = (time0+time1+time2+time3+time4)/4;

% test1 = Aeq\Beq;
% test2 = (Aeq'*Aeq)\Aeq'*Beq;
% test1-test2
f0_0 = cal_f0_sub(mem0);
f0_1 = cal_f0_sub(mem1);
f0_2 = cal_f0_sub(mem2);
f0_3 = cal_f0_sub(mem3);
f0_4 = cal_f0_sub(mem4);

plot(0:iter,f0_0)
hold on
plot(0:iter,f0_1)
plot(0:iter,f0_2)
plot(0:iter,f0_3)
plot(0:iter,f0_4)
ylabel('f_0')
xlabel('Iteration')
legend('{\alpha}_k = 0.01','{\alpha}_k = 0.001', ...
    '{\alpha}_k = 0.1','{\alpha}_k = 1/k','{\alpha}_k = 1/k^{0.5}')

xlim([0 100])
hold off

f0_star = cal_f0_sub(x');
plot(0:iter,f0_0-f0_star)
hold on
plot(0:iter,f0_1-f0_star)
plot(0:iter,f0_2-f0_star)
plot(0:iter,f0_3-f0_star)
plot(0:iter,f0_4-f0_star)
ylabel('|f_0-f_{0}^{*}|')
xlabel('Iteration')
legend('{\alpha}_k = 0.01','{\alpha}_k = 0.001', ...
    '{\alpha}_k = 0.1','{\alpha}_k = 1/k','{\alpha}_k = 1/k^{0.5}')
xlim([0 300])
hold off

function output = cal_error(mem,x_star)
    [numRows,~] = size(mem);
    temp = zeros(numRows,1);
    for k = 1:numRows
        temp(k) = mean((mem(k,:)'-x_star).^2);
        % temp(k) = sum((mem(k,:)'-x_star).^2)/128;
    end
    output = temp;
end

function output = cal_f0_sub(mem)
    [numRows,~] = size(mem);
    temp = zeros(numRows,1);
    for k = 1:numRows
        temp(k) = norm(mem(k,:)',1);
    end
    output = temp;
end