clear; clc;
%% Initialization
n_iter = 5e3; % Number of iteration of the MCMC
n_burnin = 1; % Burn-in period
n_x = 3; % Number of elements in x
% Observation data
n_particle_s = 9900; % Success in the particles
n_particle_f = 100; % Failures in the particles
n_similar_s = 4; % Success in similar products
n_similar_f = 3; % Failures in similar products
y = [n_particle_s,n_particle_f,n_similar_s,n_similar_f]'; % y is observation data, should be a column vector
% Parameters in the prior
K_U = 1000;
K_L = 500; % For K, a Gamma dist.
Alpha = 2.1;
Beta = .25; % For pi, a Beta dist.
% Setting paramters for MCMC
x_0 = [.8,(K_U+K_L)/2,Alpha/(Alpha+Beta)]'; % Inital values for x, should be a vector of n_x*1, in turns: R, K, pi
para = [K_L,K_U,Alpha,Beta]'; % Parameters for prior distributions
func_Post_K = @(x,y) Post_K(x,y,para); % Handles to posterior dist. of K
func_Post_pi = @(x,y) Post_pi(x,y,para); % Handles to posterior dist. of pi
%% MCMC
x_post = zeros(n_x,n_iter); % Allocate memory for MCMC samples
x_prev = x_0;
x_cur = x_prev;
for i = 1:n_iter
    disp([num2str(i) '/' num2str(n_iter)])
    % For R, use Gibbs sampler
    K = x_prev(2); % Extract current K
    pi = x_prev(3); % Extract current pi
    x_cur(1) = betarnd(K*pi+y(1),K*(1-pi)+y(2),1,1);
    x_prev = x_cur; 
    % For K, use MH sampler
    j = 2;
    x_cur = MCMC_MH_iter(x_prev,@Update_K,j,func_Post_K,@PropDensity_K,y,para);
    % For pi, use MH sampler
    j = 3;
    x_cur(3) = betarnd(Alpha+y(3),Beta+y(4),1,1);
    x_prev = x_cur;
    x_post(1:end,i) = x_cur; % Record the new sample
end
%% Post-processing
% Trace plot for R
figure
plot(x_post(1,1:end));
xlabel('R')
ylabel('Iterations')
title('Trace plot for R')
% Trace plot for K
figure
plot(x_post(2,1:end));
xlabel('Iterations')
ylabel('K')
title('Trace plot for K')
% Trace plot for pi
figure
plot(x_post(3,1:end));
xlabel('Iterations')
ylabel('\pi')
title('Trace plot for \pi')
% Posterior estimates for R
R_sample = x_post(1,n_burnin:end);
[P_RR,RR] = ksdensity(R_sample);
figure
plot(RR,P_RR,'k');
% Prior of pi
pi_x = 0:1e-3:1;
pi_pdf = betapdf(pi_x,Alpha,Beta);
hold on
plot(pi_x,pi_pdf,'r--')
% Posterior using only similar data
pi_pdf_post_similar = betapdf(pi_x,Alpha+y(3),Beta+y(4));
hold on
plot(pi_x,pi_pdf_post_similar,'r-')
% Posterior using mixed data
pi_pdf_post_mixed = betapdf(pi_x,Alpha+y(1)+y(3),Beta+y(2)+y(4));
hold on
plot(pi_x,pi_pdf_post_mixed,'b-')
% Posterior using only PF data
pi_pdf_post_PF_only = betapdf(pi_x,Alpha+y(1),Beta+y(2));
hold on
plot(pi_x,pi_pdf_post_PF_only,'g-')
% Labels & legends
xlabel('R')
ylabel('pdf')
title('Updated posterior pdf of R at t = 1')
legend('Using both population and monitoring data',...
    'Prior density',...
    'Using only population data',...
    'Simplified methods',...
    'Using only monitoring data')
title('Posterior density of R')
% Posterior estimates & prior for K
Ksample = x_post(2,n_burnin:end);
[pdf_KK,KK] = ksdensity(Ksample);
figure
plot(KK,pdf_KK,'k');
hold on
temp_x = linspace(K_L,K_U,1e3);
plot(temp_x,1/(K_U-K_L)*ones(1e3,1),'r--');
legend('Posterior','Prior')
title('Posterior density of K')
% Posterior estimates & prior for pi
pi_sample = x_post(3,n_burnin:end);
[pdf_pi,pipi] = ksdensity(pi_sample);
figure
plot(pipi,pdf_pi,'k');
hold on
plot(pi_x,pi_pdf,'r--');
plot(pi_x,pi_pdf_post_similar,'g-')
legend('Posterior','Prior','Posterior using only similar data')
title('Posterior density of pi')