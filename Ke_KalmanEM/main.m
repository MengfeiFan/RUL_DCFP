%% Parameter Initialization
clc; clear; close all;
mu_0 = 0;                     % mean of x_0
sigma2_0 = 0.05;              % stdev of x_0
x_0 = normrnd(mu_0,sqrt(sigma2_0));  
eta_0 = 0.5;                  % degradation rate
S_0 = 0.5;                    % shock damage
omega = 53.63;                % degradation threshold
epsilon = 0.01;               % convergence threshold for EM algorithm
T = 500;                      % the number of CM points
tau = 0.2;                    % time interval
IterMax = 500;                % maximum iteration times for EM algorithm
lambda = 0.5;                 % shock intensity
% generate degradation and shock data
C = zeros(1,T);               % indicator of arrival shocks
y = zeros(1,T);               % observed data
q = 0.7;                      % Pr of a wrong shock record
[C_pure,y,C_noise] = generate_data(lambda,x_0,eta_0,S_0,sigma2_0,T,tau,q);
C = C_pure;

%% Simulation
% seporate memory space
eta_hat = zeros(IterMax,T);   sigma2_hat = zeros(IterMax,T);    S_hat = zeros(IterMax,T);
mu_0_hat = zeros(IterMax,T);  sigma2_0_hat = zeros(IterMax,T);
x_iter = zeros(IterMax,T);    P_iter = zeros(IterMax,T);u = zeros(1,T);
Psi = zeros(IterMax,T);       % E(loglik[theta|x_0:k_hat,y_0:k(produced using theta_k,iter-1)]),obted by eq.(15)
theta = zeros(5,T);           % optimal estimates for (eta,sigma,S,mu_0,sigma2_0) at each CM point
x = zeros(1,T);               % updated x at each CM point
P = zeros(1,T);               % updated P at each CM point
% state update and parameter estimation
for k = 1:T
    iter = 0;                 % initialize the iteration indicator of EM algorithm
                              % presented by 'u' in Ke's paper
    % initialize parameters with the optimal estimates of the previous CM point
    if k == 1                 % if the current point is the first CM point
        eta_hat(iter+1,k) = eta_0;
        sigma2_hat(iter+1,k) = sigma2_0;
        S_hat(iter+1,k) = S_0;
    else
        eta_hat(iter+1,k) = theta(1,k-1);
        sigma2_hat(iter+1,k) = theta(2,k-1);
        S_hat(iter+1,k) = theta(3,k-1);
    end
    % EM iterations
    while 1
        iter = iter + 1;
        if iter > IterMax
           theta(:,k) = [eta_hat(iter-1,k);sigma2_hat(iter-1,k);S_hat(iter-1,k),mu_0_hat(iter-1,k),sigma2_0_hat(iter-1,k)];
           x(1,k) = x_iter(iter-1,k);P(1,k) = P_iter(iter-1,k);
           disp([num2str(k) 'th optimal esimates not found']);
           break
        end
        % Kalman filtering and smoothing
        A = 1;B = 1;H = 1;R = 0.001;
        if iter == 1
            if k == 1
                Q = sigma2_0*tau;
            else
                Q = theta(2,k-1)*tau;
            end
        else
            Q = sigma2_hat(iter-1,k)*tau;
        end
        % define the state equation
        if C(1,k) == 0           % without shock
            if iter == 1
                if k == 1
                    u(1,k) = tau*eta_0;
                else
                    u(1,k) = tau*theta(1,k-1);
                end
            else
                u(1,k) = tau*eta_hat(iter-1,k);
            end
        else                     % with shock
            if iter == 1
                if k == 1
                    u(1,k) = tau*eta_0 + S_0;
                else
                    u(1,k) = tau*theta(1,k-1) + theta(3,k-1);
                end
            else
                u(1,k) = tau*eta_hat(iter-1,k) + S_hat(iter-1,k);
            end
        end
        % set initial x and P
        if iter == 1
            if k == 1
                init_x = mu_0;init_P = sigma2_0;
            else
                init_x = theta(4,k-1);init_P = theta(5,k-1);
            end
        else
%             init_x = x(1,k-1);init_P = P(1,k-1); 
            init_x = mu_0_hat(iter-1,k);init_P = sigma2_0_hat(iter-1,k);
        end
        % Kalman filter
        [x_iter(iter,1:k),P_iter(iter,1:k),~,~] = kalman_filter(y(1,1:k),A,H,Q,R,init_x,init_P,'u',u(1,1:k),'B',B);
        % Kalman smoother
        [xsmooth,Psmooth,PPsmooth,~] = kalman_smoother(y(1,1:k),A,H,Q,R,init_x,init_P,'u',u(1,1:k),'B',B);
        if iter ==1
            if k == 1
                xsmooth0 = mu_0 + sigma2_0/(sigma2_0 + Q)*(xsmooth(1,1) -...
                   (1-C(1,1))*(mu_0 + eta_0*tau) - ...
                   C(1,1)*(mu_0 + eta_0*tau + S_0));
                Psmooth0 = sigma2_0 + sigma2_0^2/(sigma2_0 + Q)^2*(Psmooth(1,1) - (sigma2_0 + Q));
            else
                xsmooth0 = mu_0 + sigma2_0/(sigma2_0 + Q)*(xsmooth(1,1) -...
                   (1-C(1,1))*(mu_0 + theta(1,k-1)*tau) - ...
                   C(1,1)*(mu_0 + theta(1,k-1)*tau + theta(3,k-1)));
                Psmooth0 = sigma2_0 + sigma2_0^2/(sigma2_0 + Q)^2*(Psmooth(1,1) - (sigma2_0 + Q));               
            end
        else
            xsmooth0 = mu_0 + sigma2_0/(sigma2_0 + Q)*(xsmooth(1,1) -...
                (1-C(1,1))*(mu_0 + eta_hat(iter-1,k)*tau) - ...
                C(1,1)*(mu_0 + eta_hat(iter-1,k)*tau + S_hat(iter-1,k)));
            Psmooth0 = sigma2_0 + sigma2_0^2/(sigma2_0 + Q)^2*(Psmooth(1,1) - (sigma2_0 + Q));
        end
        % estimate parameters(mu_0,sigma_0,S,eta,sigma)using eq.(18)-(21)
        if iter == 1
            if k == 1
                [mu_0_hat(iter,k),sigma2_0_hat(iter,k),S_hat(iter,k),eta_hat(iter,k),sigma2_hat(iter,k)]...
                   = para_estimate(xsmooth,Psmooth,PPsmooth,xsmooth0,Psmooth0,S_0,eta_0,C(1,1:k),tau);
            else
                [mu_0_hat(iter,k),sigma2_0_hat(iter,k),S_hat(iter,k),eta_hat(iter,k),sigma2_hat(iter,k)]...
                   = para_estimate(xsmooth,Psmooth,PPsmooth,xsmooth0,Psmooth0,theta(3,k-1),theta(1,k-1),C(1,1:k),tau);
            end
        else
            [mu_0_hat(iter,k),sigma2_0_hat(iter,k),S_hat(iter,k),eta_hat(iter,k),sigma2_hat(iter,k)]...
                = para_estimate(xsmooth,Psmooth,PPsmooth,xsmooth0,Psmooth0,S_hat(iter-1,k),eta_hat(iter-1,k),C(1,1:k),tau);
        end
        % calculate Psi_k[theta|theta_k^(iter-1)_hat] using eq.(15)
        Psi(iter,k) = Eloglik(R,y(1,1:k),C(1,1:k),tau,xsmooth,Psmooth,PPsmooth,xsmooth0,Psmooth0,...
            eta_hat(iter,k),S_hat(iter,k),sigma2_hat(iter,k),mu_0_hat(iter,k),sigma2_0_hat(iter,k));
        % find the optimal paras estimates
        if iter == 1
            continue
        elseif abs(Psi(iter,k)/Psi(iter-1,k)-1) < epsilon
            theta(:,k) = [eta_hat(iter,k),sigma2_hat(iter,k),S_hat(iter,k),mu_0_hat(iter,k),sigma2_0_hat(iter,k)];
            x(1,k) = x_iter(iter,k);P(1,k) = P_iter(iter,k);
            disp([num2str(k) 'th iteration finished']);
            break
        else
            continue
        end
    end
end

% find the true RUL
for k = 1:T
    if y(1,k) > omega
        TTF_deg = k;
        break;
    end
end
mu_w = 1.2; % mean value of shock load
sigma_w = 0.2; % sd of shock load
D = 1.5; % threshold of hard failure
p_d = normcdf(D,mu_w,sigma_w); % probability of an arriving shock belonging to the damage zone
p_f = 1 - normcdf(D,mu_w,sigma_w); % probability of an arriving shock belonging to the fatal zone
TTF_shock = ceil(exprnd(1/(lambda*p_f/p_d))/tau);
% TTF_shock = 300;
TTF_true = min(TTF_deg,TTF_shock);
if TTF_deg<TTF_shock
    t_true = 0:TTF_deg;
    RUL_true = TTF_deg - [0:TTF_deg];
else
    t_true = 0:TTF_deg;
    diff = TTF_deg - TTF_shock;
    RUL_before = TTF_deg - [0:TTF_shock-1];
    RUL_after = zeros(1,diff+1);
    RUL_true = [RUL_before,RUL_after];
end
save('data.mat','C_pure','C_noise','TTF_shock','y');
% RUL prediction
N = T+1;                                 % the number of discrete points used to approximate the integral
L = 0:T;                                 % RUL axis
density_l = zeros(N,T);                  % pdf of RUL
mean_l = zeros(1,T);                     % mean of RUL
failure_indicator = 1;
RUL_percentile = zeros(2,T);
alpha = .05;                             % Confidence level
for k = 1:T
    for i = 1:N
        density_l(i,k) = (theta(2,k)*(omega - x(1,k)) + P(1,k)*(theta(1,k) + lambda*theta(3,k)))...
            /sqrt(2*pi*(P(1,k) + theta(2,k)*L(1,i)*tau)^3)...
            *exp(-(omega - theta(1,k)*L(1,i)*tau - lambda*theta(3,k)*L(1,i)*tau - x(1,k))^2/2/(P(1,k) + theta(2,k)*L(1,i)*tau));
        mean_l(1,k) = mean_l(1,k) + L(1,i)*density_l(i,k);
    end 
    if x(1,k)>omega
        failure_indicator = 0;
    elseif k >= TTF_shock
        failure_indicator = 0;
    end
    if failure_indicator == 0;
        mean_l(1,k) = 0;
        RUL_percentile(1,k) = 0;
        RUL_percentile(2,k) = 0; 
    else    
        if sum(density_l(:,k)) == 0
            mean_l(1,k) = T;
            RUL_percentile(1,k) = T;
            RUL_percentile(2,k) = T;
        else
            mean_l(1,k) = mean_l(1,k)/sum(density_l(:,k));
            cdf_l = cumsum(density_l(:,k))/sum(density_l(:,k));
            index_L = find(cdf_l>alpha,1);
            index_U = find(cdf_l>1-alpha,1);
            RUL_percentile(1,k) = L(1,index_L);
            RUL_percentile(2,k) = L(1,index_U);
        end
    end
end
%% Plotting
t = 1:T;
plot(t,x,'k-')
hold on
plot(t,y,'k--')
legend('Estimated value','Observed data','Location','SouthEast');
xlabel('CM point')
ylabel('x')
figure
plot(t,theta(1,:),'k-')
hold on
plot(t,eta_0*ones(1,T),'k--')
title('Estimation of \eta')
legend('Estimated value','True value');
xlabel('CM point')
ylabel('\eta')
figure
plot(t,theta(2,:),'k-')
hold on
plot(t,sigma2_0*ones(1,T),'k--')
legend('Estimated value','True value');
title('Estimation of \sigma^2')
xlabel('CM point')
ylabel('\sigma^2')
figure
plot(t,theta(3,:),'k-')
hold on
plot(t,S_0*ones(1,T),'k--')
legend('Estimated value','True value');
title('Estimation of S')
xlabel('CM point')
ylabel('S')
figure
plot(t,mean_l,'b-')
hold on
plot(t_true,RUL_true,'-k')
hold on
plot(2:T,RUL_percentile(1,2:end),'--r',...
    2:T,RUL_percentile(2,2:end),'--k')
legend('Estimated RUL','True RUL','90% belief interval lower','90% belief interval upper');
xlabel('t')
ylabel('RUL')
title('Evolution of RUL')