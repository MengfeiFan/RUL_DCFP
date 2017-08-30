function xk = sys(k,xkm1,lambda,uk)
% system function: obtain xk with xkm1
% additional parameters include:
%   a:degradation rate
%   delta_d:addtional degradation increment caused by arrived shock
%   lambda: intensity of the HPP shock process
%   tau: unit time interval
mu_w = 1.2; % mean value of shock load
sigma_w = 0.2; % sd of shock load
D = 1.8; % threshold of hard failure
p_d = normcdf(D,mu_w,sigma_w); % probability of an arriving shock belonging to the damage zone
p_f = 1 - normcdf(D,mu_w,sigma_w); % probability of an arriving shock belonging to the fatal zone
xk = zeros(3,1);
tau = 0.2;
lambda_d = lambda/p_f*p_d;
tshock = exprnd(1/lambda_d);
if tshock < tau % check out if there is a shock arriving during the current time interval
    xk(1) = xkm1(1) + xkm1(2)*tau + xkm1(3) + uk(1);
else
    xk(1) = xkm1(1) + xkm1(2)*tau + uk(1);
end
% 
xk(2) =  xkm1(2) + uk(2);
xk(3) =  xkm1(3) + uk(3);
end

