function [mu_0, sigma_0, S, eta, sigma2] = para_estimate(xsmooth, Psmooth, PPsmooth, xsmooth0, Psmooth0, S_prev, eta_prev, C, tau)
% para_estimate: Estimate parameters(mu_0,sigma_0,S,eta,sigma)using eq.(18)-(21)
% INPUTS:
% xsmooth - E[X(:,i) | y(:,1:k)]
% Psmooth - Cov[X(:,i) | y(:,1:k)] 
% PPsmooth - Cov[X(:,i), X(:,i-1) | y(:,1:k)] i >= 2
% xsmooth0 - x0|k_hat 
% Psmooth0 - P0|k_hat 
% S_prev - shock damage estimated in the previous EM iteration
% eta_prev - degradation rate estimated in the previous EM iteration
% C - a set recording the number of arrival shocks at each CM point
% tau - time interval
% OUTPUTS:
% mu_0 - updated initial mean of x
% sigma_0 - updated initial stdev of x
% S - updated shock damage
% eta - updated degradation rate
% sigma_square - updated variance of x, sigma^2

mu_0 = xsmooth0;               % estimation method not clear in Ke's paper
sigma_0 = Psmooth0;            % estimation method not clear in Ke's paper
k = length(xsmooth);
m = sum(C);                    % the number of arrival shocks
M = zeros(m,1);                % a set of time instants that shocks arrive at
count = 1;
for i = 1:k
    if C(1,i) == 1
        M(count,1) = i;
        count = count + 1;
    end
end
% estimate S using eq.(20)
if m == 0
    S = S_prev;
else
    S = 0;
    for i = 1:m
        if M(i)-1==0
            S = S + xsmooth(1,M(i))-xsmooth0-eta_prev*tau;
        else
            S = S + xsmooth(1,M(i))-xsmooth(1,M(i)-1)-eta_prev*tau;
        end
    end
    S = S/m;
end
% estimate eta using eq.(20)
eta = (xsmooth(1,k)-xsmooth(1,1)-m*S_prev)/k/tau;
% estimate sigma_square using eq.(21)
TAU = xsmooth(1,1)^2 - 2*(xsmooth(1,1)*xsmooth0 + PPsmooth(1,1))...
        + xsmooth0^2 + Psmooth(1,1) + Psmooth0;             % eq.(16)/(17)
if C(1,1) == 0                                    % if no shock arrives in the previous time interval
    sigma2 = 1/tau^2*(TAU - 2*eta_prev*tau*(xsmooth(1,1) - xsmooth0)...
        + eta_prev^2*tau^2);
else                                            % if a shock arrives in the previous time interval
    sigma2 = 1/tau^2*(TAU - 2*(eta_prev*tau + S_prev)*(xsmooth(1,1) - xsmooth0)...
        + (eta_prev*tau + S_prev)^2);
end
for i = 2:k
    TAU = xsmooth(1,i)^2 - 2*(xsmooth(1,i)*xsmooth(1,i-1) + PPsmooth(1,i))...
            + xsmooth(1,i-1)^2 + Psmooth(1,i) + Psmooth(1,i-1);             % eq.(16)/(17)
    if C(1,i) == 0                                    % if no shock arrives in the previous time interval
        sigma2 = sigma2 + 1/tau^2*(TAU - 2*eta_prev*tau*(xsmooth(1,i) - xsmooth(1,i-1))...
            + eta_prev^2*tau^2);
    else                                            % if a shock arrives in the previous time interval
        sigma2 = sigma2 + 1/tau^2*(TAU - 2*(eta_prev*tau + S_prev)*(xsmooth(1,i) - xsmooth(1,i-1))...
            + (eta_prev*tau + S_prev)^2);
    end
end
sigma2 = sigma2*tau/k;
end

