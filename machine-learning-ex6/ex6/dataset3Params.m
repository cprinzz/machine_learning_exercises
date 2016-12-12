function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


allVals = [0, 0];
bestPrediction = inf;

values = [.01, .03, .1, .3, 1, 3, 10, 30];

for c = values
  for s = values
    fprintf("C = %f, sig = %f \n",c, s);
    model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
    predictions = svmPredict(model, Xval);
    prediction = mean(double(predictions ~= yval));
    fprintf("prediction = %f \n\n", prediction);
    if prediction <= bestPrediction;
      bestPrediction = prediction;
      bestVals = [c, s];
      fprintf("New best prediction = %f \n C = %f \n sigma = %f \n\n",bestPrediction, c ,s);
    end
  end
end


C = bestVals(1);
sigma = bestVals(2);
fprintf("best C = %f \n best sigma = %f \n\n",C,sigma);







% =========================================================================

end
