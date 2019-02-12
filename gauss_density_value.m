function [prob] = gauss_density_value(mean, cov_matrix, x_var)
l = length(mean);
exp_term = ((x_var-mean).')*(inv(cov_matrix))*(x_var-mean);
prob = (1/(((2*pi)^(l/2))*(det(cov_matrix))^(1/2)))*(exp((-1)*(1/2)*exp_term));
end
