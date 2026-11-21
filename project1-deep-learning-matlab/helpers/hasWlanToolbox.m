function tf = hasWlanToolbox()
%HASWLANTOOLBOX  True if WLAN Toolbox is installed and licensed.
%
%   When it is not, the project uses its own 802.11 generator
%   (wlanPacketFallback) and an indoor multipath channel with the TGn
%   delay spreads (tgnLikeChannel) instead.

persistent cached
if isempty(cached)
    try
        wlanNonHTConfig();
        cached = true;
    catch
        cached = false;
    end
end
tf = cached;
end
