function error = prediction(x, y, mu, parameters)
%{
   Prediction

   This function computes the classification error and predicts the labels

   Input
   -----

   x: instance

   y: labels

   mu: classifier parameter

   parameters: model parameters

   Output
   ------
   
   error: classification error

%}
n_classes = parameters.n_classes;
mistakes = [];
for j = 1:length(x(:, 1))
    hat_y = predict_label(x(j, :), mu, n_classes);
    if hat_y ~= y(j) % Classification error
        mistakes(j) = 1;
    else
        mistakes(j) = 0;
    end
end
error = mean(mistakes);
end

function y = predict_label(x, mu, n_classes)
%{

   Predict

   This function assigns labels to instances

   Input
   -----

   x: instance

   mu: classifier parameter

   n_classes: number of classes

   Output
   ------

   y_pred: predicted label

%}
for j=1:n_classes
    M(j,:)=feature_vector(x', j-1, n_classes)';
    c(j) =M(j, :)*mu;
end
[~, y] = max(c);
y = y-1;
end
