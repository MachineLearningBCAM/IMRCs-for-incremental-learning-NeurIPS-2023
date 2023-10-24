function b = backward(k, stl, f, b, parameters)
%{
    Backward

    This function obtains classification rules performing forward and backward
    learning

   Input
   -----

    k: step

    stl: mean and confindence vectors single task learning

    f: mean, confidence vectors, and classifier parameters forward learning

    b: mean, confidence vectors, and classifier parameters forward and
    backward learning 

    parameters: model paramaters
    
   Output
   ------
    b: mean, confidence vectors, and classifier parameters forward and
    backward learning 
%}

m = parameters.m;
for i = 1:k
    b.d(:, i) = expected_change(k, i, parameters.w, stl.tau);
end
for i = k-1:-1:max([k-parameters.b_steps, 1])
    b = tracking_backward(k, i, parameters.m, stl, f, b);
    opt{1} = f;
    opt{2} = [];
    [prmt] = optimization('b', i, parameters, b, opt);
    b.mu(:, i) = prmt.mu;
    b.R_Ut(i) = prmt.R_Ut;
end
end

function b = tracking_backward(k, j, m, stl, f, b)
%{
﻿    Tracking_backward

     This function obtains mean vector estimates and confidence vectors

     Input
     -----

     k: step

     j: task index

     stl: mean and confidence vectors obtained at single task learning

     f: mean and confidence vectors obtained at forward learning

     b: mean and confidence vectors obtained at forward and backward learning
     
     m: length feature vector

     Output
     ------

    b: mean and confidence vectors obtained at forward and backward learning
%}

tau_backward = zeros(m , k);
s_backward = zeros(m , k);
for i = k:-1:j+1
    if i == k
        tau_backward(:, i) = stl.tau(:, k);
        s_backward(:, i) = stl.s(:, k);
    else
        for c = 1:m
            if stl.lambda(c, i)  == 0
                tau_backward(c, i) = stl.tau(c, i);
                s_backward(c, i) = 0;
            else
                tau_backward(c, i) = stl.tau(c, i) + (stl.s(c, i)/(s_backward(c, i+1) + b.d(c, i)+stl.s(c, i)))*(tau_backward(c, i+1)-stl.tau(c, i));
                s_backward(c, i) = (stl.s(c, i)^(-1) + (s_backward(c, i+1) + b.d(c, i))^(-1))^(-1);
            end
        end
    end
end
for c = 1:m
    if f.s(c, j)== 0
        b.s(c, j) = f.s(c, j);
        b.tau(c, j) = f.tau(c, j);
    else
        b.tau(c, j) = f.tau(c, j) + (f.s(c, j)/(f.s(c, j) + s_backward(c, j+1)+b.d(c, j)))*(tau_backward(c, j+1) - f.tau(c, j));
        b.s(c, j) = (f.s(c, j)^(-1) + (s_backward(c, j+1)+b.d(c, j))^(-1))^(-1);
    end
end
b.lambda(:, j) = sqrt(b.s(:, j));
end

function d_backward = expected_change(t, i, w, tau_s)
w2 = floor((w)/2);
i1 = max([1, i-w2+1]);
i2 = min([t, i+w2]);
for r = i1:i2
    if r == 1
        d_step(:, r-i1+1) = (tau_s(:, r)).^2;
    else
        d_step(:, r-i1+1) = (tau_s(:, r) - tau_s(:, r-1)).^2;
    end
end
if length(d_step(1, :))>1
    d_backward = mean(d_step');
else
    d_backward = d_step;
end
end
