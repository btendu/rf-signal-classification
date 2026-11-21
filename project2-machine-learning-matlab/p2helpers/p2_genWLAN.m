function e = p2_genWLAN(cfg)
%P2_GENWLAN  Bursty WiFi traffic on one 20 MHz channel (WLAN Toolbox).
%
%   One format per waveform: 802.11a/g OFDM, 802.11n HT, or 802.11b DSSS,
%   with random data rates, packet lengths and gaps.

fsW   = 20e6;
total = round(cfg.poolDuration*fsW);
r = rand;
if r < 0.4
    fmt = "nonHT"; bw = 16.6e6;
elseif r < 0.8
    fmt = "HT";    bw = 17.8e6;
else
    fmt = "DSSS";  bw = 16e6;
end

x = zeros(0,1);
mask = false(0,1);
while numel(x) < total
    gap = round((20e-6 + rand*180e-6)*fsW);
    switch fmt
        case "nonHT"
            mcs = randi([0 7]);
            len = randi([200 1500]);
            if p2_hasWlanToolbox()
                c = wlanNonHTConfig('ChannelBandwidth','CBW20','MCS',mcs,'PSDULength',len);
                w = wlanWaveformGenerator(randi([0 1],8*len,1), c);
            else
                w = p2_wlanPacketOwn("nonHT", mcs, len);
            end
            sub = sprintf("802.11a/g OFDM MCS %d", mcs);
        case "HT"
            mcs = randi([0 7]);
            len = randi([200 2500]);
            if p2_hasWlanToolbox()
                c = wlanHTConfig('ChannelBandwidth','CBW20','MCS',mcs,'PSDULength',len);
                w = wlanWaveformGenerator(randi([0 1],8*len,1), c);
            else
                w = p2_wlanPacketOwn("HT", mcs, len);
            end
            sub = sprintf("802.11n HT MCS %d", mcs);
        otherwise
            rates = {'1Mbps','2Mbps','5.5Mbps','11Mbps'};
            rateMbps = [1 2 5.5 11];
            ri = randi(4);
            rate = rates{ri};
            len = randi([50 400]);
            if p2_hasWlanToolbox()
                c = wlanNonHTConfig('Modulation','DSSS','DataRate',rate,'PSDULength',len);
                w = wlanWaveformGenerator(randi([0 1],8*len,1), c);
            else
                w = p2_wlanPacketOwn("DSSS", rateMbps(ri), len);
            end
            % One sample per chip at 11 MHz -> hold each chip for two
            % samples (rectangular chip pulse) -> resample 22 -> 20 MHz.
            w = resample(repelem(w(:,1),2,1), 10, 11);
            sub = "802.11b DSSS " + string(rate);
    end
    w = w(:,1);
    x    = [x; zeros(gap,1); w];                     %#ok<AGROW>
    mask = [mask; false(gap,1); true(numel(w),1)];   %#ok<AGROW>
end

[e.x, e.mask] = p2_toCommonRate(x, mask, fsW, cfg.fs);
e.bw      = bw;
e.subtype = sub;
if p2_hasWlanToolbox()
    e.source = "WLAN Toolbox wlanWaveformGenerator";
else
    e.source = "Own 802.11 generator (WLAN Toolbox not available)";
end
end
