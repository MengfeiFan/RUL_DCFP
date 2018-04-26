function p_post = Post_lambda(x,x_record,k,para)
p_post = p_deg(x,x_record,k,para)*p_lambda(x,para);  

function Likelihood_deg = p_deg(x,x_record,k,para)
tau = para(3);
sigma = para(4);
% p_f = para(5);
p_d = para(6);
deg_rate = x_record(2,:);
d = x_record(3,:);
lambda = x;
Likelihood_deg = normpdf(x_record(1,1),0,sigma); 
for i = 2:k
    mu_withshock = x_record(1,i-1) + deg_rate(1,i)*tau + d(1,i);
    mu_noshock = x_record(1,i-1) + deg_rate(1,i)*tau;
    lambda_d = lambda*p_d;
    p_shock = lambda_d*tau*exp(-lambda_d*tau);
    p_noshock = exp(-lambda_d*tau);
    Likelihood_deg = Likelihood_deg*(normpdf(x_record(1,i),mu_withshock,sigma)*p_shock + normpdf(x_record(1,i),mu_noshock,sigma)*p_noshock);
end


function p_x = p_lambda(x,para)
lambda_L = para(1);
lambda_U = para(2);
lambda = x;
p_x = pdf('Uniform',lambda,lambda_L,lambda_U);