function m = p2_metrics(yTrue, yPred, cats)
%P2_METRICS  All classification metrics listed in the handout.
%
%   m = p2_metrics(yTrue, yPred, cats)
%
%   m.accuracy, m.precision, m.recall, m.f1 (macro averages)
%   m.fpr       mean false-positive rate  FP / (FP + TN)
%   m.missRate  mean miss-detection rate  FN / (TP + FN) = 1 - recall
%   m.falseAlarmNoise   noise-only frames classified as a signal
%   m.missedAsNoise     signal frames classified as noise
%   m.perClass  table of the same per class
%   m.C         confusion matrix (rows = true, columns = predicted)

cats = cellstr(string(cats));
yTrue = categorical(cellstr(string(yTrue)), cats);
yPred = categorical(cellstr(string(yPred)), cats);
C = confusionmat(yTrue, yPred, 'Order', categorical(cats, cats));
tot = sum(C(:));
TP = diag(C);
FP = sum(C, 1)' - TP;
FN = sum(C, 2) - TP;
TN = tot - TP - FP - FN;
prec = localDiv(TP, TP + FP);
rec  = localDiv(TP, TP + FN);
f1   = localDiv(2*prec.*rec, prec + rec);
fpr  = localDiv(FP, FP + TN);
miss = localDiv(FN, TP + FN);
present = (TP + FN) > 0;           % classes that occur in the test set

m.accuracy  = sum(TP)/max(tot, 1);
m.precision = mean(prec(present));
m.recall    = mean(rec(present));
m.f1        = mean(f1(present));
m.fpr       = mean(fpr(present));
m.missRate  = mean(miss(present));
m.C = C;
m.classes = string(cats);
m.perClass = table(string(cats(:)), prec, rec, f1, fpr, miss, TP + FN, ...
    'VariableNames', ["Class" "Precision" "Recall" "F1" "FPR" "MissRate" "Support"]);

iN = find(string(cats) == "Noise", 1);
if isempty(iN) || sum(C(iN,:)) == 0
    m.falseAlarmNoise = NaN;
else
    m.falseAlarmNoise = 1 - C(iN,iN)/sum(C(iN,:));
end
if isempty(iN)
    m.missedAsNoise = NaN;
else
    others = setdiff(1:numel(cats), iN);
    m.missedAsNoise = sum(C(others, iN))/max(1, sum(C(others,:), "all"));
end
end

function r = localDiv(a, b)
r = a./b;
r(~isfinite(r)) = 0;
end
