% Return the current value of x
function x_cur = MCMC_MH_iter(x_prev,x_record,func_Update,func_Posterior,func_PropDensity,k,para)
x_cur = func_Update(x_prev,para); % Update x
func_PropDensity_override = @(x,y,para) func_PropDensity(x,y,para);
p_acc = Cal_p_acc(func_Posterior,func_PropDensity_override,x_cur,x_prev,x_record,k,para); % Calculate the acceptance prob
% Do the accept/reject
temp = rand;
if temp <= p_acc
    return;
else
    x_cur = x_prev;
    return;
end

function p_acc = Cal_p_acc(func_Posterior,func_PropDensity,x_cur,x_prev,x_record,k,para)
p_acc = func_Posterior(x_cur,x_record,k,para)/func_Posterior(x_prev,x_record,k,para)*...
    func_PropDensity(x_prev,x_cur,para)/func_PropDensity(x_cur,x_prev,para); % Normal scale
if p_acc > 1
    p_acc = 1;
end