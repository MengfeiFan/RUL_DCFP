function p = PropDensity_lambda(x,x_condition,para)
pd = handle_proposal_density_lambda(x_condition,para);
p = pdf(pd,x);