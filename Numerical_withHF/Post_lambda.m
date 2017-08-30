function p_post = Post_lambda(x,y_shock,para)
p_post = p_deg(x,y_shock,para)*p_shock(x,y_shock)*p_lambda(x,para);  

function Likelihood_deg = p_deg(x,y_shock,para)
deg = y_shock(1);
tau = para(3);
sigma = para(4);
p_f = para(5);
p_d = para(6);
deg_rate = x(2);
d = x(3);
lambda = x(4);
mu_withshock = x(1) + deg_rate*tau + d;
mu_noshock = x(1) + deg_rate*tau;
lambda_d = lambda/p_f*p_d;
p_shock = lambda_d*tau*exp(-lambda_d*tau);
p_noshock = exp(-lambda_d*tau);
Likelihood_deg = normpdf(deg,mu_withshock,sigma)*p_shock + ...
    normpdf(deg,mu_noshock,sigma)*p_noshock;

function Likelihood_shock = p_shock(x,y_shock)
t = y_shock(2);
lambda = x(4);
Likelihood_shock = exp(-lambda*t);

function p_x = p_lambda(x,para)
lambda_L = para(1);
lambda_U = para(2);
lambda = x(4);
p_x = pdf('Uniform',lambda,lambda_L,lambda_U);