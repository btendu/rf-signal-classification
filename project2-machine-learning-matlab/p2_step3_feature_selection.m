%% STEP 3 - Feature selection and optimisation
% Handout: "To reduce computational complexity, feature-selection
% techniques such as correlation analysis, mutual information, ANOVA/
% F-score, recursive feature elimination, and principal component analysis
% (PCA) can be investigated. The objective is to determine the minimum
% feature subset that provides high classification accuracy."
%
% Uses the training part of the hardest data set (level 6). For every
% ranking method, the cross-validated accuracy of a random forest is
% measured with the top-k features, k = 1, 2, 3, ... The minimum subset is
% the smallest k (over all methods) whose accuracy is within
% cfg.selectionTolerance (1 percentage point) of the best accuracy found.
%
% Output: results/p2_step3_*.png/csv and models/selection.mat

cfg = p2Config();
D = load(fullfile(cfg.dataDir, sprintf("level%d.mat", cfg.selectionLevel)));
tr = ~D.isTest;
[Xtr, st] = p2_standardize(D.X(tr, :));
ytr = D.y(tr);
names = D.featureNames;
p = numel(names);
if ~isfolder(cfg.modelDir), mkdir(cfg.modelDir); end

%% Rankings
tStart = tic;
R = p2_featureRankings(Xtr, ytr, cfg);
fprintf("Rankings computed in %.1f min.\n", toc(tStart)/60);

methods = ["Correlation" "MutualInformation" "ANOVA" "RFE"];
ranks = {R.correlation, R.mi, R.anova, R.rfe};

%% Accuracy against number of features
rng(cfg.seed, "twister");
cvp = cvpartition(ytr, "KFold", cfg.selectionFolds);
K = unique([cfg.selectionK(cfg.selectionK < p) p]);
acc = nan(numel(methods) + 1, numel(K));
for j = 1:numel(K)
    for m = 1:numel(methods)
        acc(m, j) = p2_cvAccuracy(Xtr(:, ranks{m}(1:K(j))), ytr, cvp, cfg);
    end
    acc(end, j) = p2_cvAccuracy(Xtr, ytr, cvp, cfg, true, K(j));
    fprintf("  k = %2d: %s\n", K(j), mat2str(round(100*acc(:, j)', 1)));
end
allNames = [methods "PCA"];

%% Minimum feature subset
best = max(acc(:));
target = best - cfg.selectionTolerance;
chosen = struct("method","", "k",p, "acc",acc(1,end), "idx",1:p);
for j = 1:numel(K)
    ok = find(acc(1:numel(methods), j) >= target);   % PCA still needs all features
    if ~isempty(ok)
        [~, b] = max(acc(ok, j));
        m = ok(b);
        chosen = struct("method", methods(m), "k", K(j), "acc", acc(m, j), ...
            "idx", ranks{m}(1:K(j)));
        break
    end
end
fprintf("\nBest cross-validated accuracy: %.2f %%\n", 100*best);
fprintf("Minimum subset: %d of %d features (%s), accuracy %.2f %%\n", ...
    chosen.k, p, chosen.method, 100*chosen.acc);
disp(names(chosen.idx)');

curve = array2table(acc', 'VariableNames', cellstr(allNames));
curve = addvars(curve, K', 'Before', 1, 'NewVariableNames', "NumFeatures");
writetable(curve, fullfile(cfg.resultsDir, "p2_step3_accuracy_vs_features.csv"));

top = min(15, p);
topTable = table((1:top)', names(R.correlation(1:top))', names(R.mi(1:top))', ...
    names(R.anova(1:top))', names(R.rfe(1:top))', ...
    'VariableNames', ["Rank" "Correlation" "MutualInformation" "ANOVA" "RFE"]);
writetable(topTable, fullfile(cfg.resultsDir, "p2_step3_top_features.csv"));
disp(topTable);

[~, ~, ~, ~, explained] = pca(Xtr);
nPC95 = find(cumsum(explained) >= 95, 1);
fprintf("PCA: %d components explain 95 %% of the variance.\n", nPC95);

selection = struct("featureNames", names, "rankings", R, "methods", allNames, ...
    "K", K, "acc", acc, "best", best, "chosen", chosen, "nPC95", nPC95, ...
    "explained", explained, "corrKept", R.corrKept);
save(fullfile(cfg.modelDir, "selection.mat"), "selection");

%% Figures
fig = figure("Color","w", "Position",[50 50 900 500]);
plot(K, 100*acc', "-o", "LineWidth", 1.3);
legend(allNames, "Location","southeast", "AutoUpdate","off");
hold on
yline(100*target, ":", "Best - 1 point");
xline(chosen.k, "--", sprintf("chosen: %d features", chosen.k));
xlabel("Number of features (PCA: number of components)");
ylabel("Cross-validated accuracy (%)");
title("Accuracy against number of features (level 6 training data)");
grid on
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step3_accuracy_vs_features.png"), Resolution=150);

fig = figure("Color","w", "Position",[50 50 800 700]);
imagesc(R.corr, [0 1]); axis square; colorbar;
set(gca, "XTick", 1:p, "XTickLabel", names, "YTick", 1:p, "YTickLabel", names, ...
    "FontSize", 6, "XTickLabelRotation", 90);
title(sprintf("Feature correlation |r| (%d of %d kept at |r| < %.2f)", ...
    R.corrKept, p, cfg.corrThreshold));
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step3_correlation.png"), Resolution=150);

fig = figure("Color","w", "Position",[50 50 1300 450]);
tl = tiledlayout(fig, 1, 3, "TileSpacing","compact");
ax = nexttile(tl);
[v, o] = sort(R.anovaF, "descend");
barh(ax, v(top:-1:1)); set(ax, "YTick", 1:top, "YTickLabel", names(o(top:-1:1)));
title(ax, "ANOVA F-score (top 15)");
ax = nexttile(tl);
[v, o] = sort(R.miScore, "descend");
barh(ax, v(top:-1:1)); set(ax, "YTick", 1:top, "YTickLabel", names(o(top:-1:1)));
title(ax, "Mutual information, MRMR score (top 15)");
ax = nexttile(tl);
plot(ax, cumsum(explained), "-o"); grid(ax, "on");
yline(ax, 95, ":"); xline(ax, nPC95, "--");
xlabel(ax, "Number of principal components"); ylabel(ax, "Explained variance (%)");
title(ax, sprintf("PCA: %d components for 95 %%", nPC95));
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step3_rankings.png"), Resolution=150);
