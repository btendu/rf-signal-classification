%% STEP 5 - Robustness and coexistence
% Handout: "Performance should also be evaluated as a function of SNR,
% bandwidth, frequency separation, and interference level", the
% coexistence scenarios (Bluetooth vs. Wi-Fi, ZigBee vs. Bluetooth,
% ZigBee vs. Wi-Fi, Radar vs. 5G, LTE vs. 5G) and the advanced extension
% "Comparison of shallow ML models with respect to accuracy, robustness,
% latency, and computational cost".
%
%   5a  accuracy against SNR             (level-3 models, sweep data set)
%   5b  accuracy against bandwidth       (level-3 models and test frames)
%   5c  accuracy against interference    (level-6 models and test frames)
%   5d  coexistence scenarios            (all models trained per scenario,
%                                         against frequency separation and SIR)
%   5e  final comparison of all models
%
% Output: results/p2_step5_*.csv/png and results/p2_step5_results.mat

cfg = p2Config();
R4 = load(fullfile(cfg.resultsDir, "p2_step4_results.mat"));

%% 5a - accuracy against SNR
M3 = load(fullfile(cfg.modelDir, "models_L3.mat"));
SW = load(fullfile(cfg.dataDir, "sweep.mat"));
Xs = p2_standardize(SW.X, M3.st);
snrList = cfg.sweepSNR;
accSnr = zeros(numel(cfg.modelNames), numel(snrList));
predSweep = struct();
for i = 1:numel(cfg.modelNames)
    name = cfg.modelNames(i);
    yp = p2_predictModel(M3.models.(name), Xs);
    predSweep.(name) = yp;
    for j = 1:numel(snrList)
        sel = SW.meta.sweepSNR == snrList(j);
        accSnr(i, j) = mean(yp(sel) == SW.y(sel));
    end
end
minSnr90 = nan(numel(cfg.modelNames), 1);
for i = 1:numel(cfg.modelNames)
    j = find(accSnr(i, :) >= 0.9, 1);
    if ~isempty(j), minSnr90(i) = snrList(j); end
end
snrTable = array2table(100*accSnr, 'VariableNames', "SNR_" + strrep(string(snrList), "-", "m"));
snrTable = addvars(snrTable, cfg.modelNames', minSnr90, 'Before', 1, ...
    'NewVariableNames', ["Model" "MinSNRfor90pct"]);
disp(snrTable(:, ["Model" "MinSNRfor90pct"]));
writetable(snrTable, fullfile(cfg.resultsDir, "p2_step5a_accuracy_vs_snr.csv"));

[~, bi] = max(mean(accSnr, 2));
bestSnrModel = cfg.modelNames(bi);
classRec = zeros(numel(cfg.classNames), numel(snrList));
for c = 1:numel(cfg.classNames)
    for j = 1:numel(snrList)
        sel = SW.meta.sweepSNR == snrList(j) & SW.y == cfg.classNames(c);
        classRec(c, j) = mean(predSweep.(bestSnrModel)(sel) == SW.y(sel));
    end
end

fig = figure("Color","w", "Position",[50 50 1300 480]);
tl = tiledlayout(fig, 1, 2, "TileSpacing","compact");
ax = nexttile(tl);
plot(ax, snrList, 100*accSnr', "-o", "LineWidth", 1.2);
grid(ax, "on"); ylim(ax, [0 100]); yline(ax, 90, ":");
xlabel(ax, "In-band SNR (dB)"); ylabel(ax, "Accuracy (%)");
legend(ax, cfg.modelNames, "Location","southeast", "FontSize", 7);
title(ax, "All models (level-3 conditions)");
ax = nexttile(tl);
plot(ax, snrList, 100*classRec', "-o", "LineWidth", 1.2);
grid(ax, "on"); ylim(ax, [0 100]);
xlabel(ax, "In-band SNR (dB)"); ylabel(ax, "Recall (%)");
legend(ax, cfg.classNames, "Location","southeast", "FontSize", 7);
title(ax, "Per class, " + bestSnrModel);
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step5a_accuracy_vs_snr.png"), Resolution=150);

%% 5b - accuracy against bandwidth (classes with variable bandwidth)
D3 = load(fullfile(cfg.dataDir, "level3.mat"));
Xte = p2_standardize(D3.X(D3.isTest, :), M3.st);
yte = D3.y(D3.isTest);
mt = D3.meta(D3.isTest, :);
best3 = localBest(R4.results, 3);
yp = p2_predictModel(M3.models.(best3), Xte);
bwRows = {};
for cls = ["LTE" "NR5G" "Radar" "WLAN"]
    sel = string(mt.cls) == cls;
    bwMHz = round(mt.bw(sel)/1e6, 1);
    if cls == "Radar"
        edges = [0 5 10 15 21];
        bins = discretize(mt.bw(sel)/1e6, edges);
        lab = "%g-%g MHz";
        for b = 1:numel(edges)-1
            s2 = bins == b;
            if any(s2)
                ok = yp(sel) == yte(sel);
                bwRows(end+1, :) = {cls, sprintf(lab, edges(b), edges(b+1)), nnz(s2), mean(ok(s2))}; %#ok<SAGROW>
            end
        end
    else
        u = unique(bwMHz);
        ok = yp(sel) == yte(sel);
        for b = u(:)'
            s2 = bwMHz == b;
            bwRows(end+1, :) = {cls, sprintf("%.1f MHz", b), nnz(s2), mean(ok(s2))}; %#ok<SAGROW>
        end
    end
end
bwTable = cell2table(bwRows, 'VariableNames', ["Class" "Bandwidth" "Frames" "Accuracy"]);
disp(bwTable);
writetable(bwTable, fullfile(cfg.resultsDir, "p2_step5b_accuracy_vs_bandwidth.csv"));
fig = figure("Color","w", "Position",[50 50 1000 420]);
bar(categorical(string(bwTable.Class) + " " + string(bwTable.Bandwidth), ...
    string(bwTable.Class) + " " + string(bwTable.Bandwidth)), 100*bwTable.Accuracy);
ylabel("Accuracy (%)"); ylim([0 100]); grid on
title("Accuracy against signal bandwidth (" + best3 + ", level 3)");
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step5b_accuracy_vs_bandwidth.png"), Resolution=150);

%% 5c - accuracy against interference level (level 6)
M6 = load(fullfile(cfg.modelDir, "models_L6.mat"));
D6 = load(fullfile(cfg.dataDir, "level6.mat"));
mt6 = D6.meta(D6.isTest, :);
sirEdges = [0 5 10 15 20.001];
sirBin = discretize(mt6.sirMin, sirEdges);
accSir = nan(numel(cfg.modelNames), numel(sirEdges)-1);
accNum = nan(numel(cfg.modelNames), cfg.maxInterferers);
for i = 1:numel(cfg.modelNames)
    name = cfg.modelNames(i);
    ok = M6.preds.(name) == M6.yte;
    for b = 1:numel(sirEdges)-1
        accSir(i, b) = mean(ok(sirBin == b));
    end
    for k = 1:cfg.maxInterferers
        accNum(i, k) = mean(ok(mt6.numInterf == k));
    end
end
typesList = ["adjacent" "cochannel" "adjacent+cochannel"];
best6 = localBest(R4.results, 6);
okBest = M6.preds.(best6) == M6.yte;
accType = arrayfun(@(t) mean(okBest(string(mt6.interfTypes) == t)), typesList);
sirTable = array2table(100*accSir, 'VariableNames', ["SIR_0_5" "SIR_5_10" "SIR_10_15" "SIR_15_20"]);
sirTable = addvars(sirTable, cfg.modelNames', 'Before', 1, 'NewVariableNames', "Model");
numTable = array2table(100*accNum, 'VariableNames', "Interferers_" + string(1:cfg.maxInterferers));
numTable = addvars(numTable, cfg.modelNames', 'Before', 1, 'NewVariableNames', "Model");
typeTable = table(typesList', 100*accType', 'VariableNames', ["InterferenceType" "Accuracy_" + best6]);
disp(sirTable); disp(numTable); disp(typeTable);
writetable(sirTable, fullfile(cfg.resultsDir, "p2_step5c_accuracy_vs_sir.csv"));
writetable(numTable, fullfile(cfg.resultsDir, "p2_step5c_accuracy_vs_num_interferers.csv"));
writetable(typeTable, fullfile(cfg.resultsDir, "p2_step5c_accuracy_vs_interference_type.csv"));
fig = figure("Color","w", "Position",[50 50 1300 450]);
tl = tiledlayout(fig, 1, 2, "TileSpacing","compact");
ax = nexttile(tl);
plot(ax, [2.5 7.5 12.5 17.5], 100*accSir', "-o", "LineWidth", 1.2);
grid(ax, "on"); ylim(ax, [0 100]);
xlabel(ax, "Signal-to-interference ratio of the strongest interferer (dB)");
ylabel(ax, "Accuracy (%)");
legend(ax, cfg.modelNames, "Location","southeast", "FontSize", 7);
title(ax, "Level 6: accuracy against interference level");
ax = nexttile(tl);
bar(ax, 1:cfg.maxInterferers, 100*accNum');
grid(ax, "on"); ylim(ax, [0 100]);
xlabel(ax, "Number of simultaneous interferers"); ylabel(ax, "Accuracy (%)");
title(ax, "Level 6: accuracy against number of interferers");
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step5c_interference.png"), Resolution=150);

%% 5d - coexistence scenarios
coexRows = {};
coexDetail = {};
for k = 1:size(cfg.coexPairs, 1)
    C = load(fullfile(cfg.dataDir, sprintf("coex%d.mat", k)));
    [Xtr, st] = p2_standardize(C.X(~C.isTest, :));
    Xte = p2_standardize(C.X(C.isTest, :), st);
    ytr = C.y(~C.isTest);  yte = C.y(C.isTest);
    mt = C.meta(C.isTest, :);
    cats = categories(C.y);
    models = struct();
    bestAcc = -1;
    for name = cfg.modelNames
        if name == "Hybrid"
            mdl = p2_trainModel("Hybrid", [], [], cfg, models);
        else
            mdl = p2_trainModel(name, Xtr, ytr, cfg);
        end
        models.(name) = mdl;
        yp = p2_predictModel(mdl, Xte);
        m = p2_metrics(yte, yp, cats);
        both = string(yte) == cfg.coexPairs(k,1) + "+" + cfg.coexPairs(k,2);
        detBoth = mean(yp(both) == yte(both));
        coexRows(end+1, :) = {cfg.coexPairs(k,1) + " vs " + cfg.coexPairs(k,2), ...
            name, m.accuracy, m.f1, detBoth}; %#ok<SAGROW>
        if m.accuracy > bestAcc
            bestAcc = m.accuracy; bestYp = yp; bestName = name; bestM = m;
        end
    end
    % detection of "both active" against frequency separation and SIR
    both = ~isnan(mt.overlapRatio);
    ok = bestYp == yte;
    ovEdges = [0 0.5 1 2 inf];
    ovBin = discretize(mt.overlapRatio(both), ovEdges);
    okB = ok(both);
    ov = arrayfun(@(b) mean(okB(ovBin == b)), 1:numel(ovEdges)-1);
    sEdges = [-10 -5 0 5 10.001];
    sBin = discretize(mt.sir(both), sEdges);
    sv = arrayfun(@(b) mean(okB(sBin == b)), 1:numel(sEdges)-1);
    coexDetail(end+1, :) = {cfg.coexPairs(k,1) + " vs " + cfg.coexPairs(k,2), bestName, ...
        bestAcc, ov(1), ov(2), ov(3), ov(4), sv(1), sv(2), sv(3), sv(4)}; %#ok<SAGROW>
    fig = figure("Color","w", "Position",[50 50 600 520]);
    cm = confusionchart(yte, bestYp, "Normalization","row-normalized");
    cm.Title = cfg.coexPairs(k,1) + " vs " + cfg.coexPairs(k,2) + " (" + bestName + ")";
    p2_saveFigure(fig, fullfile(cfg.resultsDir, sprintf("p2_step5d_coex%d_confusion.png", k)), Resolution=150);
    fprintf("  %s vs %s: best %s, accuracy %.3f\n", cfg.coexPairs(k,1), cfg.coexPairs(k,2), bestName, bestAcc);
end
coexTable = cell2table(coexRows, 'VariableNames', ["Scenario" "Model" "Accuracy" "MacroF1" "BothDetected"]);
coexTable.Scenario = string(coexTable.Scenario);
coexTable.Model = string(coexTable.Model);
coexDetailTable = cell2table(coexDetail, 'VariableNames', ["Scenario" "BestModel" "Accuracy" ...
    "Both_overlap_lt0p5" "Both_overlap_0p5_1" "Both_sep_1_2" "Both_sep_gt2" ...
    "Both_SIR_m10_m5" "Both_SIR_m5_0" "Both_SIR_0_5" "Both_SIR_5_10"]);
disp(coexDetailTable);
writetable(coexTable, fullfile(cfg.resultsDir, "p2_step5d_coexistence_all_models.csv"));
writetable(coexDetailTable, fullfile(cfg.resultsDir, "p2_step5d_coexistence_best.csv"));
coexWide = unstack(coexTable(:, ["Scenario" "Model" "Accuracy"]), "Accuracy", "Scenario");
fig = figure("Color","w", "Position",[50 50 1100 450]);
bar(categorical(coexWide.Model, coexWide.Model), 100*coexWide{:, 2:end});
legend(coexWide.Properties.VariableNames(2:end), "Location","southoutside", "NumColumns", 5, ...
    "Interpreter","none");
ylabel("Accuracy (%)"); ylim([0 100]); grid on
title("Coexistence scenarios: accuracy of every model");
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step5d_coexistence.png"), Resolution=150);
fig = figure("Color","w", "Position",[50 50 1200 420]);
tl = tiledlayout(fig, 1, 2, "TileSpacing","compact");
ax = nexttile(tl);
bar(ax, categorical(["<0.5 (strong overlap)" "0.5-1 (overlap)" "1-2 (adjacent)" ">2 (separate)"], ...
    ["<0.5 (strong overlap)" "0.5-1 (overlap)" "1-2 (adjacent)" ">2 (separate)"]), ...
    100*cell2mat(coexDetail(:, 4:7))');
ylabel(ax, "Both signals identified (%)"); ylim(ax, [0 100]); grid(ax, "on");
xlabel(ax, "Frequency separation / mean half-bandwidth");
legend(ax, string(coexDetail(:,1)), "Location","southoutside", "NumColumns", 3);
title(ax, "Identification of both signals against frequency separation");
ax = nexttile(tl);
plot(ax, [-7.5 -2.5 2.5 7.5], 100*cell2mat(coexDetail(:, 8:11))', "-o", "LineWidth", 1.2);
ylabel(ax, "Both signals identified (%)"); ylim(ax, [0 100]); grid(ax, "on");
xlabel(ax, "Power ratio A/B (dB)");
title(ax, "Identification of both signals against interference level");
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step5d_separation_sir.png"), Resolution=150);

%% 5e - final comparison: accuracy, robustness, latency, cost
T = load(fullfile(cfg.dataDir, "timing.mat"));
L6 = R4.results(R4.results.Level == 6, :);
cmpRows = {};
for i = 1:numel(cfg.modelNames)
    name = cfg.modelNames(i);
    r6 = L6(L6.Model == name, :);
    accAll = R4.results.Accuracy(R4.results.Model == name);
    coexMean = mean(coexTable.Accuracy(coexTable.Model == name));
    acc0 = accSnr(i, snrList == 0);
    if isempty(acc0), acc0 = NaN; end
    cmpRows(end+1, :) = {name, mean(accAll), r6.Accuracy, r6.F1, acc0, minSnr90(i), ...
        coexMean, r6.TrainTime_s, r6.InferTime_us, T.featureTimeMs*1e3 + r6.InferTime_us, ...
        r6.Size_kB}; %#ok<SAGROW>
end
comparison = cell2table(cmpRows, 'VariableNames', ["Model" "MeanAcc_L1toL6" "Acc_L6" ...
    "MacroF1_L6" "Acc_0dB" "MinSNR_90pct" "MeanAcc_Coexistence" "TrainTime_s" ...
    "Infer_us" "Total_us_per_frame" "Size_kB"]);
comparison.Model = string(comparison.Model);
comparison = sortrows(comparison, "Acc_L6", "descend");
disp(comparison);
writetable(comparison, fullfile(cfg.resultsDir, "p2_step5e_final_comparison.csv"));

fig = figure("Color","w", "Position",[50 50 800 560]);
scatter(comparison.Infer_us, 100*comparison.Acc_L6, 30 + 2*sqrt(comparison.Size_kB), "filled");
text(comparison.Infer_us*1.08, 100*comparison.Acc_L6, comparison.Model, "FontSize", 8);
set(gca, "XScale","log"); grid on
xlabel("Inference time per frame (\mus, log scale)");
ylabel("Accuracy at level 6 (%)");
title("Accuracy against latency (marker size = model size)");
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step5e_accuracy_vs_latency.png"), Resolution=150);

save(fullfile(cfg.resultsDir, "p2_step5_results.mat"), "snrTable", "accSnr", ...
    "minSnr90", "bestSnrModel", "classRec", "bwTable", "sirTable", "numTable", ...
    "typeTable", "coexTable", "coexDetailTable", "comparison");

% -------------------------------------------------------------------------
function name = localBest(results, L)
r = results(results.Level == L, :);
[~, i] = max(r.Accuracy);
name = r.Model(i);
end
