function [net, report] = buildTransferNetwork(cfg)
%BUILDTRANSFERNETWORK  Network for transfer learning from the example.
%
%   [net, report] = buildTransferNetwork(cfg)
%
%   The MathWorks "Spectrum Sensing with Deep Learning" example trains a
%   DeepLab v3+ network with a ResNet-18 backbone to label 5G NR, LTE and
%   WLAN in spectrograms. This function:
%
%     1. builds the same architecture with an output for OUR classes
%        (Noise, Bluetooth, WLAN);
%     2. copies every learned weight from the example's trained network
%        whose layer name and size match. Everything except the final
%        class-scoring layers matches, so the feature extractor starts
%        from what the example already learned about RF spectrograms;
%     3. lets the new, randomly initialised layers learn faster than the
%        copied ones.
%
%   If the example's trained network file is not found, the network still
%   starts from the ImageNet-pretrained ResNet-18 backbone. That is the
%   transfer learning the example itself performs, and "report" says which
%   of the two happened.

numClasses = numel(cfg.classNames);
net = deeplabv3plus(cfg.imageSize, numClasses, cfg.baseNetwork);

L = net.Learnables;
report = struct("source","ImageNet " + cfg.baseNetwork + " backbone only", ...
    "pretrainedFile","", "numParameters",height(L), "numCopied",0, ...
    "newLayers",strings(0,1));

preFile = localFindPretrained(cfg);
if preFile == ""
    warning("rfproject:noPretrained", ...
        "%s not found. Starting from the ImageNet backbone only. " + ...
        "See README, section 'Getting the example network'.", cfg.pretrainedFile);
    return
end

pre = localLoadNetwork(preFile);
if isempty(pre)
    warning("rfproject:badPretrained", ...
        "Could not read a dlnetwork from %s. Starting from the ImageNet backbone only.", preFile);
    return
end

% ---- Copy learnable parameters with the same layer, name and size.
P = pre.Learnables;
[found, loc] = ismember(L.Layer + "/" + L.Parameter, P.Layer + "/" + P.Parameter);
values = L.Value;
copied = false(height(L),1);
for i = 1:height(L)
    if found(i)
        v = P.Value{loc(i)};
        if isequal(size(v), size(values{i}))
            values{i} = dlarray(gather(extractdata(v)));
            copied(i) = true;
        end
    end
end
net.Learnables.Value = values;

% ---- Copy batch-normalisation statistics the same way.
S = net.State;
Q = pre.State;
if ~isempty(S) && ~isempty(Q)
    [found, loc] = ismember(S.Layer + "/" + S.Parameter, Q.Layer + "/" + Q.Parameter);
    for i = 1:height(S)
        if found(i) && isequal(size(Q.Value{loc(i)}), size(S.Value{i}))
            S.Value{i} = dlarray(gather(extractdata(Q.Value{loc(i)})));
        end
    end
    net.State = S;
end

% ---- New layers learn faster. Layers whose rate is fixed at 0 (the
% bilinear upsampling layers) are left alone.
newIdx = find(~copied);
for i = newIdx(:).'
    try
        f = getLearnRateFactor(net, L.Layer(i), L.Parameter(i));
        if f > 0
            net = setLearnRateFactor(net, L.Layer(i), L.Parameter(i), ...
                f*cfg.headLearnRateBoost);
        end
    catch
        % Not every parameter exposes a learn-rate factor; skip those.
    end
end

report.pretrainedFile = string(preFile);
report.numCopied      = nnz(copied);
report.newLayers      = unique(L.Layer(~copied));
if report.numCopied == 0
    report.source = "ImageNet " + cfg.baseNetwork + " backbone only";
    warning("rfproject:nothingCopied", ...
        "No weights in %s matched this network. Starting from the ImageNet backbone only.", preFile);
else
    report.source = "Spectrum Sensing example network";
end
end

% -------------------------------------------------------------------------
function f = localFindPretrained(cfg)
f = "";
candidate = fullfile(cfg.pretrainedDir, cfg.pretrainedFile);
if isfile(candidate)
    f = string(candidate);
    return
end
onPath = which(cfg.pretrainedFile);
if ~isempty(onPath)
    f = string(onPath);
end
end

% -------------------------------------------------------------------------
function pre = localLoadNetwork(preFile)
pre = [];
S = load(preFile);
names = fieldnames(S);
for i = 1:numel(names)
    v = S.(names{i});
    if isa(v,"dlnetwork")
        pre = v;
        return
    end
end
% Networks saved by older releases are DAGNetwork objects.
for i = 1:numel(names)
    v = S.(names{i});
    if isa(v,"DAGNetwork")
        try
            pre = dag2dlnetwork(v);
        catch
            pre = [];
        end
        return
    end
end
end
