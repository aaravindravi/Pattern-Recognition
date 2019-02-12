%% 2 - class mahalonobis_classifier
function [class, distance1, distance2] = mahalonobis_class(mean1, mean2, cov_matrix1,cov_matrix2, x_var)
for samples = 1:size(x_var,2)

distance1(samples,1) = sqrt((x_var(:,samples)-mean1).'*inv(cov_matrix1)*(x_var(:,samples)-mean1));
distance2(samples,1) = sqrt((x_var(:,samples)-mean2).'*inv(cov_matrix2)*(x_var(:,samples)-mean2));

if distance1(samples,1) < distance2(samples,1)
    class(samples,1)=1;
else
    class(samples,1)=2;
end
end