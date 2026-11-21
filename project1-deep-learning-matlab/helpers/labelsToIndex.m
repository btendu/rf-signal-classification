function idx = labelsToIndex(labels, cfg)
%LABELSTOINDEX  Convert a label image to class numbers 1..numClasses.
%
%   idx = labelsToIndex(labels, cfg)
%
%   labels may be a categorical array (output of semanticseg or of a pixel
%   label datastore) or a numeric array of the IDs in cfg.labelIDs.
%   Pixels with no valid class get 0.

if iscategorical(labels)
    [~, idx] = ismember(string(labels), cfg.classNames);
else
    [~, idx] = ismember(double(labels), cfg.labelIDs);
end
end
