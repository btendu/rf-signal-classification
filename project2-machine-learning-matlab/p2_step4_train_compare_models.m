%% STEP 4 - Train and compare the ML models (all six impairment levels)
% Handout: "Proposed ML models: SVM, Random Forest, KNN, Decision Tree,
% Naive Bayes, Logistic Regression, Gradient Boosting / XGBoost, AdaBoost"
% and "Performance Evaluation: accuracy, precision, recall, F1-score,
% confusion matrix, false-positive rate, miss-detection rate, training
% time, prediction/inference time, model size, computational complexity."
% Advanced extension: "Hybrid ML ensemble: RF + XGBoost + SVM".
%
% For every level 1-6 every model is trained on 75 % of the frames and
% tested on the other 25 %. At level 6 the models are also retrained with
% only the minimum feature subset of step 3.
%
% Output: results/p2_step4_*.csv/png and models/models_L1..L6.mat

cfg = p2Config();
if ~isfolder(cfg.modelDir), mkdir(cfg.modelDir); end
baseNames = setdiff(cfg.modelNames, "Hybrid", "stable");
rows = {};
for L = 1:6
    D = load(fullfile(cfg.dataDir, sprintf("level%d.mat", L)));
    [Xtr, st] = p2_standardize(D.X(~D.isTest, :));
    Xte = p2_standardize(D.X(D.isTest, :), st);
    ytr = D.y(~D.isTest);
    yte = D.y(D.isTest);
    cats = categories(D.y);
    models = struct();
    preds = struct();
    for name = cfg.modelNames
        if name == "Hybrid"
            mdl = p2_trainModel("Hybrid", [], [], cfg, models);
        else
            mdl = p2_trainModel(name, Xtr, ytr, cfg);
        end
        [yp, ~, tPred] = p2_predictModel(mdl, Xte);
        m = p2_metrics(yte, yp, cats);
        kb = p2_modelSizeKB(mdl);
        models.(name) = mdl;
        preds.(name) = yp;
        rows(end+1, :) = {L, name, m.accuracy, m.precision, m.recall, m.f1, ...
            m.fpr, m.missRate, m.falseAlarmNoise, m.missedAsNoise, ...
            mdl.trainTime, tPred*1e6, kb}; %#ok<SAGROW>
        fprintf("  L%d %-18s acc %.3f  F1 %.3f  train %6.2f s  predict %7.1f us/frame  %8.0f kB\n", ...
            L, name, m.accuracy, m.f1, mdl.trainTime, tPred*1e6, kb);
    end
    featureNames = D.featureNames;
    save(fullfile(cfg.modelDir, sprintf("models_L%d.mat", L)), ...
        "models", "st", "cats", "featureNames", "preds", "yte");
end
results = cell2table(rows, 'VariableNames', ["Level" "Model" "Accuracy" ...
    "Precision" "Recall" "F1" "FPR" "MissRate" "FalseAlarmNoise" ...
    "MissedAsNoise" "TrainTime_s" "InferTime_us" "Size_kB"]);
results.Model = string(results.Model);
writetable(results, fullfile(cfg.resultsDir, "p2_step4_results_all_levels.csv"));

%% Accuracy table: models x levels
accTable = unstack(results(:, ["Level" "Model" "Accuracy"]), "Accuracy", "Level", ...
    'NewDataVariableNames', "L" + string(1:6));
disp(accTable);
writetable(accTable, fullfile(cfg.resultsDir, "p2_step4_accuracy_by_level.csv"));
L6 = results(results.Level == 6, :);
[~, ib] = max(L6.Accuracy);
bestModel = L6.Model(ib);
fprintf("Best model at level 6: %s (accuracy %.3f)\n", bestModel, L6.Accuracy(ib));

%% Minimum feature subset vs all features (level 6)
S = load(fullfile(cfg.modelDir, "selection.mat"));
idx = S.selection.chosen.idx;
D = load(fullfile(cfg.dataDir, "level6.mat"));
[Xtr, st] = p2_standardize(D.X(~D.isTest, idx));
Xte = p2_standardize(D.X(D.isTest, idx), st);
ytr = D.y(~D.isTest);  yte = D.y(D.isTest);
cats = categories(D.y);
subRows = {};
modelsSub = struct();
for name = cfg.modelNames
    if name == "Hybrid"
        mdl = p2_trainModel("Hybrid", [], [], cfg, modelsSub);
    else
        mdl = p2_trainModel(name, Xtr, ytr, cfg);
    end
    modelsSub.(name) = mdl;
    [yp, ~, tPred] = p2_predictModel(mdl, Xte);
    m = p2_metrics(yte, yp, cats);
    allRow = L6(L6.Model == name, :);
    subRows(end+1, :) = {name, allRow.Accuracy, m.accuracy, allRow.InferTime_us, ...
        tPred*1e6, allRow.Size_kB, p2_modelSizeKB(mdl)}; %#ok<SAGROW>
end
subsetTable = cell2table(subRows, 'VariableNames', ["Model" "Acc_AllFeatures" ...
    "Acc_Subset" "Infer_us_All" "Infer_us_Subset" "Size_kB_All" "Size_kB_Subset"]);
fprintf("\nAll %d features vs minimum subset of %d (%s):\n", numel(D.featureNames), ...
    numel(idx), S.selection.chosen.method);
disp(subsetTable);
writetable(subsetTable, fullfile(cfg.resultsDir, "p2_step4_all_vs_subset.csv"));

%% Export level-6 features for the optional Python XGBoost check
T = array2table(D.X, 'VariableNames', cellstr(D.featureNames));
T.label = string(D.y);
T.isTest = D.isTest;
writetable(T, fullfile(cfg.dataDir, "level6_features.csv"));

save(fullfile(cfg.resultsDir, "p2_step4_results.mat"), "results", "accTable", ...
    "bestModel", "subsetTable");

%% Figures
fig = figure("Color","w", "Position",[50 50 1100 480]);
A = accTable{:, 2:end};
bar(categorical(accTable.Model, accTable.Model), 100*A);
ylabel("Test accuracy (%)"); ylim([0 100]); grid on
legend(cfg.levelNames, "Location","southoutside", "NumColumns", 3);
title("Accuracy of every model at every impairment level");
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step4_accuracy_by_level.png"), Resolution=150);

M = load(fullfile(cfg.modelDir, "models_L6.mat"));
fig = figure("Color","w", "Position",[50 50 1500 1300]);
tl = tiledlayout(fig, 3, 3, "TileSpacing","compact");
title(tl, "Confusion matrices at level 6 (multiple simultaneous signals)");
for name = cfg.modelNames
    nexttile(tl);
    cm = confusionchart(M.yte, M.preds.(name), "Normalization","row-normalized");
    cm.Title = name;
    cm.FontSize = 7;
end
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step4_confusion_L6.png"), Resolution=150);

fig = figure("Color","w", "Position",[50 50 700 600]);
cm = confusionchart(M.yte, M.preds.(bestModel), "Normalization","row-normalized");
cm.Title = "Best model at level 6: " + bestModel;
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step4_confusion_best_L6.png"), Resolution=150);

fig = figure("Color","w", "Position",[50 50 1000 450]);
tl = tiledlayout(fig, 1, 2, "TileSpacing","compact");
ax = nexttile(tl);
bar(ax, categorical(L6.Model, L6.Model), L6.TrainTime_s);
set(ax, "YScale","log"); ylabel(ax, "Training time (s)"); grid(ax, "on");
title(ax, "Training time (level 6)");
ax = nexttile(tl);
bar(ax, categorical(L6.Model, L6.Model), L6.InferTime_us);
set(ax, "YScale","log"); ylabel(ax, "Inference time per frame (\mus)"); grid(ax, "on");
title(ax, "Inference time (level 6)");
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step4_timing.png"), Resolution=150);
