function [Psi] = Eloglik(R, y, C, tau, xsmooth, Psmooth, PPsmooth, xsmooth0, Psmooth0, eta, S, sigma2, mu_0, sigma2_0)
% Eloglik: Calculate Psi_k[theta|theta_k^(iter-1)_hat] using eq.(15)
% INPUTS:
% R - the observation covariance
% y - the measurement data
% C - a set recording the number of arrival shocks at each CM point
% tau - time interval
% xsmooth - E[X(:,i) | y(:,1:k)]
% Psmooth - Cov[X(:,i) | y(:,1:k)]
% PPsmooth - Cov[X(:,i), X(:,i-1) | y(:,1:k)] i >= 2
% xsmooth0 - x0|k_hat 
% Psmooth0 - P0|k_hat 
% eta - the degradation rate
% S - shock damage
% sigma - stdev of x
% mu_0 - initial mean of x
% sigma_0 - initial stdev of x
% OUTPUTS:
% Psi - Psi_k[theta|theta_k^(iter-1)_hat] calculated by eq.(15)

k = length(xsmooth);
% calculate the 1st part in eq.(15)
part1 = 0;
for i = 1:k
    part1 = part1 + log(R) + (y(1,i)^2 - 2*y(1,i)*xsmooth(1,i) + xsmooth(1,i)^2 + Psmooth(1,i))/R; 
end
% calculate the 2nd and 3rd parts in eq.(15)
TAU = xsmooth(1,1)^2 - 2*(xsmooth(1,1)*xsmooth0 + PPsmooth(1,1))...
        + xsmooth0^2 + Psmooth(1,1) + Psmooth0;             % eq.(16)/(17)
if C(1,1) == 0                                    % if no shock arrives in the previous time interval
    part23 = log(sigma2*tau) + (TAU - 2*eta*tau*(xsmooth(1,1)-xsmooth0) + eta^2*tau^2)/(sigma2*tau);
else                                            % if a shock arrives in the previous time interval
    part23 = log(sigma2*tau) + (TAU - 2*(eta*tau + S)*(xsmooth(1,1)-xsmooth0) + (eta*tau + S)^2)/(sigma2*tau);
end
for i = 2:k
    TAU = xsmooth(1,i)^2 - 2*(xsmooth(1,i)*xsmooth(1,i-1) + PPsmooth(1,i))...
            + xsmooth(1,i-1)^2 + Psmooth(1,i) + Psmooth(1,i-1);             % eq.(16)/(17)
    if C(1,i) == 0                                    % if no shock arrives in the previous time interval
        part23 = part23 + log(sigma2*tau) + (TAU - 2*eta*tau*(xsmooth(1,i)-xsmooth(1,i-1)) + eta^2*tau^2)/(sigma2*tau);
    else                                            % if a shock arrives in the previous time interval
        part23 = part23 + log(sigma2*tau) + (TAU - 2*(eta*tau + S)*(xsmooth(1,i)-xsmooth(1,i-1)) + (eta*tau + S)^2)/(sigma2*tau);
    end 
end
% calculate the 4th part in eq.(15)
part4 = log(sigma2_0);
% calculate the 5th part in eq.(15)
part5 = (xsmooth0^2 - 2*mu_0*xsmooth0 + mu_0^2 + Psmooth0)/sigma2_0;
% calculate Psi
Psi = - part1 - part23 - part4 - part5;
end

