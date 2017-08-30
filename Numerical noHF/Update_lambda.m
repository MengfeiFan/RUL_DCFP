function x_cur = Update_lambda(x_prev,index,para)
x_cur = x_prev;
handle_prop = handle_proposal_density_lambda(x_prev(index),para);
x_cur(index) = random(handle_prop); % Update the index-th element of x