function R = p2_featureRankings(X, y, cfg)
%P2_FEATURERANKINGS  Rank features with the handout's selection methods.
%
%   R = p2_featureRankings(X, y, cfg)       X standardised training features
%
%   R.anova        ANOVA F-score (between-class / within-class variance)
%   R.mi           mutual information, minimum-redundancy (fscmrmr)
%   R.correlation  ANOVA order, then drop features correlated > threshold
%                  with an already kept one (correlation analysis)
%   R.rfe          recursive feature elimination with random-forest
%                  permutation importance
%   Each field holds feature indices, best first, plus R.anovaF, R.miScore,
%   R.corr (the correlation matrix).

p = size(X, 2);

% ---- ANOVA F-score
F = zeros(1, p);
g = double(y);
K = max(g);
n = numel(g);
for j = 1:p
    xj = X(:, j);
    mAll = mean(xj);
    ssb = 0; ssw = 0;
    for c = 1:K
        v = xj(g == c);
        if isempty(v), continue; end
        ssb = ssb + numel(v)*(mean(v) - mAll)^2;
        ssw = ssw + sum((v - mean(v)).^2);
    end
    F(j) = (ssb/(K - 1))/max(ssw/(n - K), eps);
end
[~, R.anova] = sort(F, "descend");
R.anovaF = F;

% ---- mutual information (minimum redundancy, maximum relevance)
[R.mi, miScore] = fscmrmr(X, y);
R.miScore = miScore;

% ---- correlation filter
Cr = abs(corr(X));
Cr(~isfinite(Cr)) = 0;
R.corr = Cr;
kept = [];
dropped = [];
for j = R.anova
    if isempty(kept) || all(Cr(j, kept) < cfg.corrThreshold)
        kept(end+1) = j; %#ok<AGROW>
    else
        dropped(end+1) = j; %#ok<AGROW>
    end
end
R.correlation = [kept dropped];
R.corrKept = numel(kept);

% ---- recursive feature elimination
remaining = 1:p;
eliminated = [];
while numel(remaining) > 1
    t = templateTree('NumVariablesToSample', max(1, round(sqrt(numel(remaining)))), ...
        'Reproducible', true);
    mdl = fitcensemble(X(:, remaining), y, 'Method','Bag', ...
        'NumLearningCycles', cfg.selectionTrees, 'Learners', t);
    imp = oobPermutedPredictorImportance(mdl);
    nDrop = max(1, round(0.1*numel(remaining)));
    [~, order] = sort(imp, "ascend");
    drop = remaining(order(1:nDrop));
    eliminated = [eliminated drop]; %#ok<AGROW>
    remaining = setdiff(remaining, drop, "stable");
end
R.rfe = [remaining fliplr(eliminated)];
end
