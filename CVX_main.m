clc
clear
close all
load('cs.mat')

addpath(genpath([pwd filesep 'cvx']))

% F_us, sampling mask, X_us are known
F_us0 = F_us(sampling_mask==1,:);
X_us0 = X_us(sampling_mask==1);

F_us0_real = real(F_us0);
F_us0_imag = imag(F_us0);
X_us0_real = real(X_us0);
X_us0_imag = imag(X_us0);

F0 = [F_us0_real;F_us0_imag];
X0 = [X_us0_real;X_us0_imag];


[numRows_x,~] = size(x);

cvx_begin
    variable x_hat0(numRows_x)
    minimize (norm(x_hat0,1))
    subject to
        F0*x_hat0 == X0
cvx_end

lambda1 = 0.001
cvx_begin
    variable x_hat1(numRows_x)
    minimize (norm(F0*x_hat1-X0,2) + lambda1*norm(x_hat1,1))
cvx_end

stem(x_hat1,'^')
hold on
stem(x_hat0,'x')
stem(x)
legend('x\_hat (l1 norm)','x\_hat (Lasso)','x')
title('Compressed sensing solved by CVX solver')
xlim([0 128])
xlabel('time')
hold off

cal_error(x_hat0',x)
cal_error(x_hat1',x)

function output = cal_error(mem,x_star)
    [numRows,~] = size(mem);
    temp = zeros(numRows,1);
    for k = 1:numRows
        temp(k) = mean((mem(k,:)'-x_star).^2);
        % temp(k) = sum((mem(k,:)'-x_star).^2)/128;
    end
    output = temp;
end