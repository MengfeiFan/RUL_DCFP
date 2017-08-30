function [ C,y,C_noise ] = generate_data( lambda,x_0,eta,S,sigma2,T,tau,q )
%generate_data: Generate degradation and shock data 
% INPUTS:
% lambda - shock intensity
% x_0 - initial degradation level
% eta - degradation rate
% S - shock damage
% sigma2 - factor of Brown motion
% T - the number of CM point
% tau - time interval
% q - Pr of a wrong shock record
% OUTPUTS:
% C - shock indicator set
% y - measurement (degradation) data

C = zeros(1,T);
C_noise = zeros(1,T);        % shock set with random inverse
y = zeros(1,T);
y_0 = x_0;
dW = zeros(1,T);             % increaments A
randn('state',100)           % set the state of randn
for i = 1:T
    dW(1,i) = sqrt(sigma2*tau)*randn;
end
p = lambda*tau;              % Pr of a shock arrives in a time interval
for i = 1:T
    rndnum = unifrnd(0,1);
    if i == 1
        y_prev = y_0;
    else
        y_prev = y(1,i-1);
    end
    if rndnum < p
        C(1,i) = 1;
        y(1,i) = y_prev + tau*eta + dW(1,i) + S;
    else
        y(1,i) = y_prev + tau*eta + dW(1,i);
    end
end
for i = 1:T
    rndnum = unifrnd(0,1);
    if rndnum < q
        C_noise(1,i) = 1 - C(1,i);
    else
        C_noise(1,i) = C(1,i);
    end
end
end

