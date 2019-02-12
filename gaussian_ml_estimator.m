%% Gaussian Maximum Likelihood Estimator
function [m_hat, s_hat] = gaussian_ml_estimator(X)
s_hat = zeros(size(X,2),size(X,2));
N = length(X);
m_hat = ((1/N)*(sum(X))).';
for index = 1:N
s_hat = s_hat + (((X(index,:).')-m_hat)*((X(index,:).')-m_hat).');
end
s_hat = (1/N) * s_hat;
end