function [x, spans] = genWlanFrame(cfg)
%GENWLANFRAME  One frame of bursty WLAN traffic on a single 20 MHz channel.
%
%   [x, spans] = genWlanFrame(cfg)
%
%   x      N-by-1 complex single vector at cfg.wlanRate (20 MHz), baseband.
%   spans  one row per packet: [firstSample lastSample occupiedBandwidthHz].
%
%   Packets are a random mix of 802.11a/g OFDM, 802.11n HT and 802.11b
%   DSSS/CCK (the format most 2.4 GHz access points use for beacons), with
%   random lengths, data rates and idle gaps.

fs = cfg.wlanRate;
N  = round(cfg.frameDuration*fs);
x     = complex(zeros(N,1,"single"));
spans = zeros(0,3);

% One "how busy is this network" value per frame.
maxIdle = cfg.wlanIdleRange(1) + rand*diff(cfg.wlanIdleRange);

pos = 1 + round(rand*maxIdle*fs);
while pos <= N - round(100e-6*fs)
    [w, bw] = localRandomPacket(cfg);
    n1 = min(N, pos + numel(w) - 1);
    x(pos:n1) = single(w(1:n1-pos+1));
    spans(end+1,:) = [pos n1 bw]; %#ok<AGROW>
    pos = n1 + 1 + round((50e-6 + rand*maxIdle)*fs);
end
end

% -------------------------------------------------------------------------
function [w, bw] = localRandomPacket(cfg)
p = cumsum(cfg.wlanFormatProb)/sum(cfg.wlanFormatProb);
r = rand;
if r < p(1)
    % 802.11a/g OFDM: 52 subcarriers, about 16.6 MHz wide
    mcs = randi([0 7]);  len = randi([100 2300]);
    if hasWlanToolbox()
        c = wlanNonHTConfig(ChannelBandwidth="CBW20", MCS=mcs, PSDULength=len);
        w = wlanWaveformGenerator(randi([0 1],8*c.PSDULength,1), c);
    else
        w = wlanPacketFallback("nonHT", mcs, len);
    end
    bw = 16.6e6;
elseif r < p(2)
    % 802.11n HT: 56 subcarriers, about 17.8 MHz wide
    mcs = randi([0 7]);  len = randi([200 4000]);
    if hasWlanToolbox()
        c = wlanHTConfig(ChannelBandwidth="CBW20", MCS=mcs, PSDULength=len);
        w = wlanWaveformGenerator(randi([0 1],8*c.PSDULength,1), c);
    else
        w = wlanPacketFallback("HT", mcs, len);
    end
    bw = 17.8e6;
else
    % 802.11b DSSS/CCK. The generator returns one sample per chip at
    % 11 MHz. Holding each chip for two samples gives the rectangular chip
    % pulse (main lobe +/-11 MHz); resampling 22 MHz -> 20 MHz brings it to
    % the common WLAN rate.
    rates = ["1Mbps" "2Mbps" "5.5Mbps" "11Mbps"];
    rateMbps = [1 2 5.5 11];
    ri = randi(numel(rates));  len = randi([40 400]);
    if hasWlanToolbox()
        c = wlanNonHTConfig(Modulation="DSSS", DataRate=rates(ri), PSDULength=len);
        w = wlanWaveformGenerator(randi([0 1],8*c.PSDULength,1), c);
    else
        w = wlanPacketFallback("DSSS", rateMbps(ri), len);
    end
    w  = repelem(w(:,1), 2, 1);
    w  = resample(w, 10, 11);
    bw = 16e6;
end
w = double(w(:,1));
end
