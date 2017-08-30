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
sigma_u = [1e-3;1e-3;1e-3];                        % sd of x; degradation rate; delta_d
p_sys_noise   = @(u) normpdf(u, 0, sigma_u);
gen_sys_noise = @(u) normrnd(0, sigma_u);          % sample from p_sys_noise (returns column vector)

% PDF of observation noise and noise generator function
nv = 1;                                           % size of the vector of observation noise
sigma_v = 2e-2;
p_obs_noise   = @(v) normpdf(v, 0, sigma_v);
gen_obs_noise = @(v) normrnd(0, sigma_v);         % sample from p_obs_noise (returns column vector)

% Initial PDF
% p_x0 = @(x) normpdf(x, 0,sqrt(10));             % initial pdf
gen_x0 = @(x) [0; random('unif',1.5e-3,5.5e-3); random('unif',6e-2,7.7e-2)];              % sample from p_x0 (returns column vector)

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
D = 1.8; % threshold of hard failure
p_d = normcdf(D,mu_w,sigma_w); % probability of an arriving shock belonging to the damage zone
p_f = 1 - normcdf(D,mu_w,sigma_w); % probability of an arriving shock belonging to the fatal zone
tau = 10; % time interval

% Observation data
% y_shock = [x_obs,t_current]'; % y is observation data, should be a column vector
% Parameters in the prior
lambda_U = 0.249/p_d*p_f;
lambda_L = 0.248/p_d*p_f; % For lambda, a Uniform dist.
% Setting paramters for MCMC
lambda_0 = (lambda_U+lambda_L)/2; % Inital values for lambda
para = [lambda_L,lambda_U,tau,sigma_v,p_f,p_d]'; % Parameters for prior distributions
func_Post_lambda = @(x,y_shock) Post_lambda(x,y_shock,para); % Handles to posterior dist. of lambda

% simulation
x_post = zeros(n_iter,1); % Allocate memory for MCMC samples
x_prev = [0;3.5e-3;6.9e-2;lambda_0];

%% State Updating and Parameter Estimation
% Number of time steps
T = 20;

% Separate memory space
x = zeros(nx,T);  y = zeros(ny,T);
u = zeros(nu,T);  v = zeros(nv,T);
% y_true = y;

% Simulate system
xh0 = [0;3.5e-3;6.9e-2];                    % initial state
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

load('weardata.mat')
y = wear;
y = y';
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
   %[xh(:,k), pf] = particle_filter(sys, y(:,k), pf, 'multinomial_resampling');
   [xh(:,k), pf] = particle_filter(@sys, lambda_set, y(:,k), pf, 'systematic_resampling');  
   % filtered observation
   yh(:,k) = obs(k, xh(:,k), 0);
   
   for i = 1:n_iter
%         disp([num2str(i) '/' num2str(n_iter)])
        % For R, use Gibbs sampler
        x_prev = [xh(:,k);lambda_mean(k-1)];  
        y_shock = [y(:,k),(k-1)*tau]';
        j = 4;
        x_cur = MCMC_MH_iter(x_prev,@Update_lambda,j,func_Post_lambda,@PropDensity_lambda,y_shock,para);
        x_post(i) = x_cur(4); % Record the new sample
   end
   lambda_set = x_post;
   lambda_mean(k) = mean(lambda_set);
   lambda_record(:,k) = x_post;
end
lambda_d = lambda_mean*p_d/p_f;
%% RUL prediction
y_th = 0.7; % Failure threshold

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
MaxIter = 10*T;
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

% Determine the credibility interval
RUL_percentile = zeros(2,T);
RUL_sort = zeros(Ns,T);
w_k_sort = zeros(Ns,T);
alpha = .05; % Confidence level
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
alpha = .05; % Confidence level
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

%% Plotting
% Plot x VS t
figure
plot(time,xh(1,:),'sb-')
hold on
plot(time,y,'ok-')
hold on
plot(time,xh_percentile(1,1:end),'--r',...
    time,xh_percentile(2,1:end),'--k')
legend('Estimated x','Observation data','90% belief interval lower','90% belief interval upper');
xlabel('t')
ylabel('x')
title('Evolution of degradation')
% Plot RUL VS t
figure
plot(2:T,sum(RUL(:,2:T).*sample_w(:,2:T)),'sb-')
hold on
plot(t_plot,RUL_plot,'-ok')
hold on
plot(2:T,RUL_percentile(1,2:end),'--r',...
    2:T,RUL_percentile(2,2:end),'--k')
legend('Estimated RUL','True RUL','90% belief interval lower','90% belief interval upper');
xlabel('t')
ylabel('RUL')
title('Evolution of RUL')
% plot estimated parameters
figure
plot(time,xh(2,:))
xlabel('CM point')
ylabel('\eta')
figure
plot(time,xh(3,:))
xlabel('CM point')
ylabel('S')
figure
plot(2:T,lambda_d(2:end),'k-')
title('Estimation of \lambda')
xlabel('CM point')
ylabel('\lambda')