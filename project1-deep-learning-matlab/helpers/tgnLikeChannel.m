function [y, modelName] = tgnLikeChannel(x, fs)
%TGNLIKECHANNEL  Indoor multipath fading without WLAN Toolbox.
%
%   [y, modelName] = tgnLikeChannel(x, fs)
%
%   Used only when WLAN Toolbox (wlanTGnChannel) is not available. Picks
%   one of the IEEE 802.11n (TGn) indoor models A-E by its RMS delay
%   spread (0, 15, 30, 50, 100 ns) and builds an exponential power-delay
%   profile with 10 ns tap spacing, faded with comm.RayleighChannel
%   (Communications Toolbox). The TGn clustering and line-of-sight
%   components are not modelled.

names = ["Model-A" "Model-B" "Model-C" "Model-D" "Model-E"];
rms   = [0 15 30 50 100]*1e-9;
k = randi(numel(names));
modelName = names(k);

if rms(k) == 0
    delays = 0;
    gains  = 0;
else
    delays = 0:10e-9:5*rms(k);
    gains  = -10*log10(exp(1))*delays/rms(k);   % exponential decay in dB
end
chan = comm.RayleighChannel('SampleRate',fs, 'PathDelays',delays, ...
    'AveragePathGains',gains, 'NormalizePathGains',true, ...
    'MaximumDopplerShift',3);
y = chan(double(x(:)));
end
