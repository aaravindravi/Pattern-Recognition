function [pc] = bayes_class(mean,cov,prob,X)

for index = 1:size(mean,3) %No. of classes
    for samples = 1:size(X,2)
        pxw(samples,index) = gauss_density_value(mean(:,:,index),cov(:,:,index),X(:,samples));

        pc(samples,index) = pxw(samples,index)*prob(:,index);
    end
end
