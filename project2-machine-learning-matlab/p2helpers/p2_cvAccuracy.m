function acc = p2_cvAccuracy(X, y, cvp, cfg, usePCA, k)
%P2_CVACCURACY  Cross-validated random-forest accuracy on a feature set.
%
%   acc = p2_cvAccuracy(X, y, cvp, cfg)              all columns of X
%   acc = p2_cvAccuracy(X, y, cvp, cfg, true, k)     first k principal
%                                                    components (PCA fitted
%                                                    on each training fold)

if nargin < 5, usePCA = false; end
nOK = 0;
nAll = 0;
for f = 1:cvp.NumTestSets
    tr = training(cvp, f);
    te = test(cvp, f);
    Xtr = X(tr, :);
    Xte = X(te, :);
    if usePCA
        [coeff, ~, ~, ~, ~, mu] = pca(Xtr);
        k2 = min(k, size(coeff, 2));
        Xtr = (Xtr - mu)*coeff(:, 1:k2);
        Xte = (Xte - mu)*coeff(:, 1:k2);
    end
    t = templateTree('NumVariablesToSample', max(1, round(sqrt(size(Xtr,2)))), ...
        'Reproducible', true);
    mdl = fitcensemble(Xtr, y(tr), 'Method','Bag', ...
        'NumLearningCycles', cfg.selectionTrees, 'Learners', t);
    yp = predict(mdl, Xte);
    nOK = nOK + sum(yp == y(te));
    nAll = nAll + nnz(te);
end
acc = nOK/nAll;
end
