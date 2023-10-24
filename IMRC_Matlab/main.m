%%% Load dataset
dataset_name = '';
load(dataset_name)

%%%%% Input parameters
parameters.batch_size = 10; % Samples per task, -1 if we use all the samples
parameters.n_classes = 2; % Number of classes
parameters.b_steps = 3; % Number of backward steps
parameters.w = 2; % Window size
parameters.max_iter = 2000; % Maximum number of iterations in the optimization
%%%%%
disp('Starting IMRC')
%%%%%
parameters.m = parameters.n_classes*(length(X_train{1}(1, :))+1); % Length of the feature vector

%%% Number of tasks
n_tasks = length(X_train);

%%% Initialization of mean and confidence vectors
[stl, f, b] = initialize();

%%% Start running 
for k = 1:n_tasks

    disp(['Learning task ',num2str(k)])

    %%% Single task learning
    stl = single_task(k, X_train{k}, Y_train{k}, stl, parameters);

    %%% Forward learning
    f = forward(k, stl, f, parameters);

    %%% Forward and backward learning
    b = backward(k, stl, f, b, parameters);

    %%% Prediction

    %%% Test set
    x_test = X_test{k};
    y_test = Y_test{k};

    %%% Single task learning
    error_s_tasks(k) = prediction(x_test, y_test, stl.mu(:, k), parameters);

    %%% Forward learning
    error_f_tasks(k) = prediction(x_test, y_test, f.mu(:, k), parameters);

    %%% Forward and backward learning
    err_b(k, k) = error_f_tasks(k);
    for i = k-1:-1:max([k-parameters.b_steps, 1])
        x_test = X_test{i};
        y_test = Y_test{i};
        err_b(i, k) = prediction(x_test, y_test, b.mu(:, i), parameters);
    end
end
for i = 1:parameters.b_steps
    for k = 1:n_tasks
        if k>n_tasks - i
            error_b_tasks(i, k) = err_b(k, end);
        else
            error_b_tasks(i, k) = err_b(k, k+i-1);
        end
    end
end
error_f = mean(error_f_tasks);
error_s = mean(error_s_tasks);
error_b = mean(error_b_tasks');

disp(['The classification error of IMRC method is ', num2str(error_b(parameters.b_steps))])
clearvars -except error_f error_s error_f_tasks error_s_tasks error_b_tasks error_b
save results
