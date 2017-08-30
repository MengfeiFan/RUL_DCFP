% Return the current value of x
function x_cur = MCMC_MH_iter(x_prev,func_Update,index,func_Posterior,func_PropDensity,y,para)
x_cur = func_Update(x_prev,index,para); % Update the index-th element of x
func_PropDensity_override = @(x,y) func_PropDensity(x,y,para);
p_acc = Cal_p_acc(func_Posterior,func_PropDensity_override,x_cur,x_prev,y,index); % Calculate the acceptance prob
% Do the accept/reject
temp = rand;
if temp <= p_acc
    return;
else
    x_cur = x_prev;
    return;
end

function p_acc = Cal_p_acc(func_Posterior,func_PropDensity,x_cur,x_prev,y,index)
p_acc = func_Posterior(x_cur,y)/func_Posterior(x_prev,y)*...
    func_PropDensity(x_prev(index),x_cur(index))/func_PropDensity(x_cur(index),x_prev(index)); % Normal scale
% log_p_cc = func_Posterior(x_cur,y)-func_Posterior(x_prev,y)+...
%     func_PropDensity(x_prev(index),x_cur(index))-func_PropDensity(x_cur(index),x_prev(index)); % Log scale
% p_acc = exp(log_p_cc);
if p_acc > 1
    p_acc = 1;
end