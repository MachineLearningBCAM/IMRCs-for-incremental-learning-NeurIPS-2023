function stl = single_task(k, X_train, Y_train, stl, parameters)
%{
﻿    Single task learning
     This function obtains classification rules performing single task
     learning
     Input
     -----
     k: step
     single_variables: mean and confidence vectors obtained at single task
     learning
     x_train and y_train: sample set
     parameters: model parameters
     Output
     ------
     stl: mean and confidence vectors and classifier parameter obtained at single task
     learning
%}
Feature = [];
if parameters.batch_size < 0
    xt = X_train;
    yt = Y_train;
else
    c = cvpartition(Y_train,'Holdout',parameters.batch_size,'Stratify',true);
    idx = test(c);
    xt = X_train(idx, :);
    yt = Y_train(idx);
end
for i = 1:length(xt(:, 1))
    Feature(i, :) = feature_vector(xt(i, :)', yt(i), parameters.n_classes);
end
stl.tau(:, k) = mean(Feature);
stl.s(:, k) = var(Feature)./(length(Feature(:, 1)));
stl.lambda(:, k) = sqrt(var(Feature)./(length(Feature(:, 1))));
opt{1} = [];
opt{2} = xt;
stl.x = xt;
[prmt] = optimization('s', k, parameters, stl, opt);
stl.mu(:, k) = prmt.mu;
stl.R_Ut(:, k) = prmt.R_Ut;
end
