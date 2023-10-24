function phi=feature_vector(x,y, n_classes)
%{
   Feature_vector

   This function obtains feature vectors

   Input
   -----

   x: new instance

   y: new label

   n_classes: number of classes

   Output
   ------

   phi: feature vector

%}
x_phi = [1; x];
e = zeros(n_classes, 1);
e(y+1) = 1;
phi = kron(e, double(x_phi));
end
