function model = p2_trainModel(name, X, y, cfg, components)
%P2_TRAINMODEL  Train one of the handout's classifiers on standardised features.
%
%   model = p2_trainModel(name, X, y, cfg)
%   model = p2_trainModel("Hybrid", [], [], cfg, components)
%
%   name: "SVM" "RandomForest" "KNN" "DecisionTree" "NaiveBayes"
%         "LogisticRegression" "GradientBoosting" "AdaBoost" "Hybrid"
%
%   X must already be standardised (p2_standardize). y is categorical.
%   The Hybrid ensemble (handout: RF + XGBoost + SVM) averages the class
%   probabilities of the trained RandomForest, GradientBoosting and SVM
%   models passed in "components" (a struct with those three fields).
%
%   model.name, model.mdl, model.classes, model.trainTime (s)
%
%   Gradient boosting: MATLAB has no XGBoost, so gradient-boosted trees
%   are built with LogitBoost (boosting with the logistic loss, i.e.
%   gradient boosting of trees) in a one-vs-all scheme. An optional
%   Python script (p2_xgboost_optional.py) runs real XGBoost on the same
%   features for comparison.

model.name = string(name);
if model.name == "Hybrid"
    model.mdl = {components.RandomForest.mdl, components.GradientBoosting.mdl, ...
                 components.SVM.mdl};
    model.classes = components.SVM.classes;
    model.trainTime = components.RandomForest.trainTime + ...
        components.GradientBoosting.trainTime + components.SVM.trainTime;
    return
end

classes = categories(y);
model.classes = classes;
p = size(X, 2);
tStart = tic;
switch model.name
    case "SVM"
        t = templateSVM('KernelFunction','gaussian', 'KernelScale','auto');
        mdl = fitcecoc(X, y, 'Learners',t, 'Coding','onevsone', ...
            'FitPosterior',true, 'ClassNames',classes);
    case "RandomForest"
        t = templateTree('NumVariablesToSample', max(1, round(sqrt(p))), ...
            'Reproducible',true);
        mdl = fitcensemble(X, y, 'Method','Bag', ...
            'NumLearningCycles',cfg.numTrees, 'Learners',t, 'ClassNames',classes);
    case "KNN"
        mdl = fitcknn(X, y, 'NumNeighbors',cfg.knnK, 'ClassNames',classes);
    case "DecisionTree"
        mdl = fitctree(X, y, 'ClassNames',classes);
    case "NaiveBayes"
        % A tiny jitter avoids "zero variance" errors for features that are
        % constant inside one class (e.g. a count that is always 0 for noise).
        Xj = X + 1e-6*randn(size(X));
        mdl = fitcnb(Xj, y, 'ClassNames',classes);
    case "LogisticRegression"
        t = templateLinear('Learner','logistic', 'Regularization','ridge', ...
            'Lambda',1e-4);
        mdl = fitcecoc(X, y, 'Learners',t, 'Coding','onevsall', 'ClassNames',classes);
    case "GradientBoosting"
        t = templateEnsemble('LogitBoost', cfg.numBoost, ...
            templateTree('MaxNumSplits',cfg.boostSplits), 'LearnRate',0.1);
        try
            mdl = fitcecoc(X, y, 'Learners',t, 'Coding','onevsall', ...
                'FitPosterior',true, 'ClassNames',classes);
        catch
            % older releases: no posterior fitting for ensemble learners
            mdl = fitcecoc(X, y, 'Learners',t, 'Coding','onevsall', ...
                'ClassNames',classes);
        end
    case "AdaBoost"
        mdl = fitcensemble(X, y, 'Method','AdaBoostM2', ...
            'NumLearningCycles',cfg.numBoost, ...
            'Learners',templateTree('MaxNumSplits',cfg.boostSplits), ...
            'LearnRate',0.1, 'ClassNames',classes);
    otherwise
        error("p2:unknownModel", "Unknown model %s.", name);
end
model.trainTime = toc(tStart);
try
    model.mdl = compact(mdl);
catch
    model.mdl = mdl;          % KNN has no compact form
end
end
