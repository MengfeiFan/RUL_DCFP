function handle_prop = handle_proposal_density_lambda(x_index,para)
lambda_L = para(1);
lambda_U = para(2);
handle_prop = makedist('uniform',lambda_L,lambda_U);
return;        