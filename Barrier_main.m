clc
clear
close all

%% Data set init

load('cs.mat');

F_us_real = real(F_us(sampling_mask == 1,:));
F_us_imag = imag(F_us(sampling_mask == 1,:));

X_us_real = real(X_us(sampling_mask == 1));
X_us_imag = imag(X_us(sampling_mask == 1));

F = [F_us_real;F_us_imag];
X = [X_us_real;X_us_imag];

p=length(x);
x_initial = pinv(F'*F)*F'*X;
%beta = X\y;
%beta = ones(p,1);

global newton_vals;
newton_vals = [];

mu=20;

lambda = 0.001;
%delta0=1/lambda;
t=1000/lambda;

u=(max(x_initial)+1)*ones(p,1);
%t = 2*beta;
x_hat=[x_initial;u];

initial_val=objval(F,X,lambda,p,t,x_hat);

global obj_val;
global obj_it;
global x_tmp;

obj_val = initial_val;
obj_it = 1;

x_tmp = x_hat(1:p);

tic;
[opt_x,opt_value,inner_it]=barrier(F,X,lambda,p,t,x_hat,1e-4,mu);
elapsed_time = toc;
disp(['CPU time:' num2str(elapsed_time) 's']);
opt_x = opt_x(1:p);
opt_x_thr = opt_x;
opt_x_thr(opt_x_thr < 1e-3) = 0;
for i = 1:length(x_tmp(1,:))
    MSE(i) = mean((x_tmp(:,i)-x).^2);
end
figure(1)
plot(obj_it,obj_val,'LineWidth', 1.5);
%set(gca,'XMinorTick','on','YMinorTick','on');
hold on;
plot(obj_it,obj_val,'r*');
xlabel({'Outer Iterations with \mu=20'}) 
ylabel('f_0 - f_0^*') 
xticks(round(obj_it));
for i=1:size(obj_it,2)
    tmp =  sprintf('%0.3f', obj_val(i));
    text(obj_it(i),obj_val(i),['(',(num2str(tmp)),')'],'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left','color','b');
end

figure(2)
plot(newton_vals,'LineWidth', 1.5);
xline(748, 'r--', 'LineWidth', 1.5);
xlabel({'Inner Iterations with \mu=20'}) 
ylabel('Duality Gap') 

figure(3)
plot(MSE,'LineWidth', 1.5);
xlabel({'Barrier Method Iterations';'(outer loop)(\mu=20)'}) 
ylabel('MSE') 
figure(4)
scatter(1:p,opt_x,'o','DisplayName','Recovered x','MarkerEdgeColor','r');
hold on
scatter(1:p,x,'x','DisplayName','Ground truth', 'MarkerEdgeColor','b');
legend('show')
grid on;
xlabel({'Barrier Method Iterations';'(outer loop)(\mu=20)'}) 
ylabel('x') 
hold off



