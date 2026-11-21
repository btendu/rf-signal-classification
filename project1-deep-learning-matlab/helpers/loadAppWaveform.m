function [waveform, fs] = loadAppWaveform(file, defaultFs)
%LOADAPPWAVEFORM  Read a waveform saved by the Wireless Waveform Generator.
%
%   [waveform, fs] = loadAppWaveform(file, defaultFs)
%
%   Works with a MAT-file exported from the app and with the files written
%   by step1_generate_waveforms. The file is searched (including inside
%   structs) for the first complex vector, which is taken as the waveform,
%   and for a scalar named Fs or SampleRate. If no sample rate is stored,
%   defaultFs is used and a warning says so.

S = load(file);
[waveform, fs] = localSearch(S);
if isempty(waveform)
    error("rfproject:noWaveform", "No complex waveform found in %s.", file);
end
if isempty(fs)
    warning("rfproject:noSampleRate", ...
        "No sample rate found in %s. Assuming %.3g MHz.", file, defaultFs/1e6);
    fs = defaultFs;
end
waveform = double(waveform(:,1));
end

% -------------------------------------------------------------------------
function [w, fs] = localSearch(S)
w  = [];
fs = [];
rateNames = ["fs" "samplerate" "samplingrate"];
names = fieldnames(S);
for i = 1:numel(names)
    v = S.(names{i});
    if isstruct(v) && isscalar(v)
        [w2, fs2] = localSearch(v);
        if isempty(w),  w  = w2;  end
        if isempty(fs), fs = fs2; end
    elseif isnumeric(v) && ~isreal(v) && numel(v) > 100
        if isempty(w)
            w = v;
            if size(w,1) == 1
                w = w.';
            end
        end
    elseif isnumeric(v) && isscalar(v) && isreal(v) ...
            && any(lower(string(names{i})) == rateNames)
        if isempty(fs)
            fs = double(v);
        end
    end
end
end
