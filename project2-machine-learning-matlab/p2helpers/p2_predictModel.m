function [yhat, scores, secondsPerFrame] = p2_predictModel(model, X)
%P2_PREDICTMODEL  Class predictions (and scores) of a p2_trainModel model.
%
%   [yhat, scores, secondsPerFrame] = p2_predictModel(model, X)
%
%   scores: class probabilities where the model provides them (SVM,
%   RandomForest, GradientBoosting, NaiveBayes, KNN, DecisionTree, Hybrid);
%   for the other models the raw classification scores.

cats = model.classes;
tStart = tic;
switch model.name
    case {"SVM", "GradientBoosting"}
        [yhat, scores] = localEcocPosterior(model.mdl, X);
    case "LogisticRegression"
        [yhat, scores] = predict(model.mdl, X);
    case "Hybrid"
        [~, pRF] = predict(model.mdl{1}, X);
        [~, pGB]  = localEcocPosterior(model.mdl{2}, X);
        [~, pSVM] = localEcocPosterior(model.mdl{3}, X);
        scores = (pRF + pGB + pSVM)/3;
        [~, k] = max(scores, [], 2);
        yhat = categorical(cats(k), cats);
    otherwise
        [yhat, scores] = predict(model.mdl, X);
end
secondsPerFrame = toc(tStart)/max(1, size(X,1));
yhat = categorical(cellstr(string(yhat)), cats);
end

% -------------------------------------------------------------------------
function [yhat, P] = localEcocPosterior(mdl, X)
% Posterior probabilities of an ECOC model; if posteriors were not fitted,
% a softmax of the negative losses is used instead.
try
    [yhat, ~, ~, P] = predict(mdl, X);
catch
    [yhat, negLoss] = predict(mdl, X);
    z = negLoss - max(negLoss, [], 2);
    P = exp(5*z);
    P = P./sum(P, 2);
end
if isempty(P) || any(~isfinite(P(:)))
    [yhat, negLoss] = predict(mdl, X);
    z = negLoss - max(negLoss, [], 2);
    P = exp(5*z)./sum(exp(5*z), 2);
end
end
