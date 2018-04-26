function x_cur = Update_lambda(x_prev,para)
% x_cur = x_prev;
handle_prop = handle_proposal_density_lambda(x_prev,para);
x_cur = random(handle_prop); % Update the index-th element of x