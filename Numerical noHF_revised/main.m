%% clear memory, screen, and close all figures
clear, clc, close all;

%% Particle Filtering Initialization
% Process equation x[k] = sys(k, x[k-1], u[k]);
nx = 3;  % number of states

% Observation equation y[k] = obs(k, x[k], v[k]);
ny = 1;                                         % number of observations
obs = @(k, xk, vk) xk(1) + vk;                  % (returns column vector)

% PDF of process noise and noise generator function
nu = 3;                                            % size of the vector of process noise
sigma_u = [sqrt(5e-2*0.2);1e-2;1e-2];                 % sd of x; degradation rate; delta_d
p_sys_noise   = @(u) normpdf(u, 0, sigma_u);
gen_sys_noise = @(u) normrnd(0, sigma_u);          % sample from p_sys_noise (returns column vector)

% PDF of observation noise and noise generator function
nv = 1;                                           % size of the vector of observation noise
sigma_v = 1e-2;
p_obs_noise   = @(v) normpdf(v, 0, sigma_v);
gen_obs_noise = @(v) normrnd(0, sigma_v);         % sample from p_obs_noise (returns column vector)

% Initial PDF
% p_x0 = @(x) normpdf(x, 0,sqrt(10));             % initial pdf
gen_x0 = @(x) [normrnd(0,sqrt(0.05)); random('unif',0.4,0.6); random('unif',0.4,0.6)];               % sample from p_x0 (returns column vector)

% Transition prior PDF p(x[k] | x[k-1])
% (under the suposition of additive process noise)
% p_xk_given_xkm1 = @(k, xk, xkm1) p_sys_noise(xk - sys(k, xkm1, 0));

% Observation likelihood PDF p(y[k] | x[k])
% (under the suposition of additive process noise)
p_yk_given_xk = @(k, yk, xk) p_obs_noise(yk - obs(k, xk, 0));

%% MCMC Initialization
n_iter = 5e2; % Number of iteration of the MCMC
n_burnin = 1; % Burn-in period

mu_w = 1.2; % mean value of shock load
sigma_w = 0.2; % sd of shock load
D = 1.5; % threshold of hard failure
p_d = normcdf(D,mu_w,sigma_w); % probability of an arriving shock belonging to the damage zone
p_f = 1 - normcdf(D,mu_w,sigma_w); % probability of an arriving shock belonging to the fatal zone
tau = 0.2; % time interval

% Observation data
% y_shock = [x_obs,t_current]'; % y is observation data, should be a column vector
% Parameters in the prior
lambda_U = 0.8/p_d;
lambda_L = 0.2/p_d; % For lambda, a Uniform dist.
% Setting paramters for MCMC
lambda_0 = 0.5/p_d; % Inital values for lambda
para = [lambda_L,lambda_U,tau,sigma_u(1),p_f,p_d]'; % Parameters for prior distributions
func_Post_lambda = @(x,x_record,k,para) Post_lambda(x,x_record,k,para); % Handles to posterior dist. of lambda

% simulation
lambda_post = zeros(n_iter,1); % Allocate memory for MCMC samples
lambda_prev = lambda_0;

%% State Updating and Parameter Estimation
% Number of time steps
T = 500;

% Separate memory space
x = zeros(nx,T);  y = zeros(ny,T);
u = zeros(nu,T);  v = zeros(nv,T);
% y_true = y;

% Simulate system
xh0 = [0;0.5;0.5];                    % initial state
u(:,1) = 0;                               % initial process noise
v(:,1) = gen_obs_noise(sigma_v);          % initial observation noise
x(:,1) = xh0;
% y(:,1) = obs(1, xh0, v(:,1));
% y_true(:,1) = obs(1, xh0, 0);
for k = 2:T
    % here we are basically sampling from p_xk_given_xkm1 and from p_yk_given_xk
    u(:,k) = gen_sys_noise();              % simulate process noise
    v(:,k) = gen_obs_noise();              % simulate observation noise
    x(:,k) = sys(k, x(:,k-1), lambda_0, u(:,k));     % simulate state
    %    y(:,k) = obs(k, x(:,k),   v(:,k));     % simulate observation
    %    y_true(:,k) = obs(k, x(:,k), 0);
end

load('data.mat') % observed degradation data and shock records
y_true = y;

% Separate memory
xh = zeros(nx, T); xh(:,1) = xh0;
yh = zeros(ny, T); yh(:,1) = obs(1, xh0, 0);
lambda_set = lambda_0*ones(n_iter,1);
lambda_mean = lambda_0*ones(T,1);
lambda_record = zeros(n_iter,T);

pf.k               = 1;                   % initial iteration number
pf.Ns              = n_iter;                 % number of particles
pf.w               = zeros(pf.Ns, T);     % weights
pf.particles       = zeros(nx, pf.Ns, T); % particles
pf.gen_x0          = gen_x0;              % function for sampling from initial pdf p_x0
pf.p_yk_given_xk   = p_yk_given_xk;       % function of the observation likelihood PDF p(y[k] | x[k])
pf.gen_sys_noise   = gen_sys_noise;       % function for generating system noise
%pf.p_x0 = p_x0;                          % initial prior PDF p(x[0])
%pf.p_xk_given_ xkm1 = p_xk_given_xkm1;   % transition prior PDF p(x[k] | x[k-1])

% Estimate state
for k = 2:T
    fprintf('Iteration = %d/%d\n',k,T);
    % state estimation
    pf.k = k;
    [xh(:,k), pf] = particle_filter(@sys, lambda_post, y(:,k), pf, 'systematic_resampling');
    % filtered observation
    yh(:,k) = obs(k, xh(:,k), 0);
    
    lambda_prev = lambda_mean(k-1);
    for i = 1:n_iter
        lambda_post(i) = MCMC_MH_iter(lambda_prev,xh,@Update_lambda,func_Post_lambda,@PropDensity_lambda,k,para); % Record the new sample
        lambda_prev = lambda_post(i);
    end
    lambda_mean(k) = mean(lambda_post);
    lambda_record(:,k) = lambda_post;
end
lambda_d = lambda_mean*p_d;
%% RUL prediction
y_th = 53.63; % Failure threshold

% Find true TTF
for i = 1:T
    if y_true(i) > y_th
        TTF_deg = i;
        break;
    end
end

N = n_iter; % the number of hard failure time samples
TTF_shock = 10000;
t_plot = [0:TTF_deg];
RUL_plot = TTF_deg - [0:TTF_deg];

time = [1:T]; % Time instants that used to make prediciton
sample_particles = pf.particles; % Particles
sample_w = pf.w; % Weights of each particle at each t
Ns = pf.Ns; % Number of particles
RUL = zeros(Ns,T); % Time to failure
RUL_shock = zeros(Ns,T); % Time to failure
MaxIter = 2*T;

% Prediction
for i = 2:T
    fprintf('i = %d / %d\n',i,T)
    t = time(i);
    sample_para = sample_particles(:,:,t); % Estimated xs by particles at each t
    for j = 1:Ns
        xkm = sample_para(:,j);
        if i>=TTF_shock
            RUL(j,i) = 0;
        elseif obs(t,xkm,0) > y_th % if the current time has already failed
            RUL(j,i) = 0;
        else
            % Search for the TTF
            k = t+1;
            while 1
                xk_pred = sys(k,xkm,lambda_mean(i),[0;0;0]);
                yk_pred = obs(k,xk_pred,0);
                if yk_pred > y_th
                    RUL(j,i) = k-t;
                    break;
                else
                    xkm = xk_pred;
                    k = k+1;
                end
                if k == MaxIter
                    RUL(j,i) = MaxIter;
                    break;
                end
            end
        end
    end
end

% Determine the confidence intervals
alpha = .05; % Confidence level

RUL_percentile = zeros(2,T);
RUL_sort = zeros(Ns,T);
w_k_sort = zeros(Ns,T);
for i = 2:T
    [RUL_sort(:,i), I] = sort(RUL(1:end,i));
    temp_w = sample_w(:,time(i));
    w_k_sort(:,i) = temp_w(I);
    RUL_k_cdf = cumsum(w_k_sort(:,i));
    index_L = find(RUL_k_cdf>alpha,1);
    index_U = find(RUL_k_cdf>1-alpha,1);
    RUL_percentile(1,i) = RUL_sort(index_L,i);
    RUL_percentile(2,i) = RUL_sort(index_U,i);
end

xh_percentile = zeros(2,T);
xh_sort = zeros(Ns,T);
w_xh_sort = zeros(Ns,T);
for i = 2:T
    [xh_sort(:,i), I_xh] = sort(sample_particles(1,1:end,i));
    temp_w = sample_w(:,time(i));
    w_xh_sort(:,i) = temp_w(I_xh);
    xh_cdf = cumsum(w_xh_sort(:,i));
    index_L = find(xh_cdf>alpha,1);
    index_U = find(xh_cdf>1-alpha,1);
    xh_percentile(1,i) = xh_sort(index_L,i);
    xh_percentile(2,i) = xh_sort(index_U,i);
end

lambda_percentile = zeros(2,T);
lambda_sort = zeros(n_iter,T);
eta_percentile = zeros(2,T);
eta_sort = zeros(Ns,T);
S_percentile = zeros(2,T);
S_sort = zeros(Ns,T);
for i = 2:T
    lambda_sort(:,i) = sort(lambda_record(1:end,i));
    lambda_percentile(1,i) = lambda_sort(25,i)*p_d;
    lambda_percentile(2,i) = lambda_sort(475,i)*p_d;
    
    [eta_sort(:,i), I_eta] = sort(sample_particles(2,1:end,i));
    temp_w = sample_w(:,time(i));
    w_eta_sort(:,i) = temp_w(I_eta);
    eta_cdf = cumsum(w_eta_sort(:,i));
    eta_index_L = find(eta_cdf>alpha,1);
    eta_index_U = find(eta_cdf>1-alpha,1);
    eta_percentile(1,i) = eta_sort(eta_index_L,i);
    eta_percentile(2,i) = eta_sort(eta_index_U,i);
    
    [S_sort(:,i), I_S] = sort(sample_particles(3,1:end,i));
    w_S_sort(:,i) = temp_w(I_S);
    S_cdf = cumsum(w_S_sort(:,i));
    S_index_L = find(S_cdf>alpha,1);
    S_index_U = find(S_cdf>1-alpha,1);
    S_percentile(1,i) = S_sort(S_index_L,i);
    S_percentile(2,i) = S_sort(S_index_U,i);
end

%% Plotting
% Plot x VS t
figure
plot(time,xh(1,:),'k-')
hold on
plot(time,y,'k--')
hold on
legend('Estimated value','Observation data','Location','SouthEast');
xlabel('Measurement point')
ylabel('x')

% Plot predicted RUL and 90% confidence intervals
figure
plot(2:T,sum(RUL(:,2:T).*sample_w(:,2:T)),'k--')
hold on
plot(t_plot,RUL_plot,'k-')
hold on
plot(2:T,RUL_percentile(1,2:end),'k:',...
    2:T,RUL_percentile(2,2:end),'k:')
legend('Estimated value','True value','90% confidence interval');
xlabel('Measurement point')
ylabel('RUL')

% plot estimated parameters and 90% confidence intervals
figure
plot(time,xh(2,:),'k-')
hold on
plot(time,0.5*ones(1,T),'k--')
hold on
plot(2:T,eta_percentile(1,2:end),'k:',...
    2:T,eta_percentile(2,2:end),'k:')
legend('Estimated value','True value','90% confidence interval');
xlabel('Measurement point')
ylabel('\eta')
axis([0 500 0 1.6])

figure
plot(time,xh(3,:),'k-')
hold on
plot(time,0.5*ones(1,T),'k--')
hold on
plot(2:T,S_percentile(1,2:end),'k:',...
    2:T,S_percentile(2,2:end),'k:')
legend('Estimated value','True value','90% confidence interval');
xlabel('Measurement point')
ylabel('S')
axis([0 500 -0.4 0.75])

figure
plot(time,lambda_d/p_d,'k-')
hold on
plot(time,0.53/p_d*ones(1,T),'k--')
hold on
plot(2:T,lambda_percentile(1,2:end)/p_d,'k:',...
    2:T,lambda_percentile(2,2:end)/p_d,'k:')
legend('Estimated value','True value','90% confidence interval');
xlabel('Measurement point')
ylabel('\lambda')
axis([0 500 0 1/p_d])

% Trace plot
figure
t_trace = 200; % choose the time point
plot(lambda_record(:,t_trace),'k-'); % the trace plot at t_tracexlabel('Measurement point')
hold on
plot(time(1:500),0.53/p_d*ones(1,500),'k--')
xlabel('MCMC sample')
ylabel('\lambda')
title('MCMC trace plot at t_2_0_0')
legend('MCMC sample','True value');
axis([0 500 0 1/p_d])
figure
t_trace = 450; % choose the time point
plot(lambda_record(:,t_trace),'k-'); % the trace plot at t_trace
hold on
plot(time(1:500),0.53/p_d*ones(1,500),'k--')
xlabel('MCMC sample')
ylabel('\lambda')
title('MCMC trace plot at t_4_5_0')
legend('MCMC sample','True value');
axis([0 500 0 1/p_d])