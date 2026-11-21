function e = p2_genNR(cfg)
%P2_GENNR  5G NR downlink waveform (5G Toolbox).
%
%   Random numerology and bandwidth: 15 kHz (5, 10, 20 MHz) or 30 kHz
%   (10, 20 MHz) subcarrier spacing, full-band PDSCH with QPSK to 256-QAM.
%   If the custom configuration is rejected by your release, the default
%   nrDLCarrierConfig is used; without 5G Toolbox an NR-numerology OFDM
%   signal is generated instead. e.source says which happened.

scs  = [15 15 15 30 30];          % kHz
bwM  = [5 10 20 10 20];           % MHz
nrb  = [25 52 106 24 51];
mods = {'QPSK','16QAM','64QAM','256QAM'};
k = randi(numel(scs));
mo = mods{randi(numel(mods))};

if exist("nrWaveformGenerator","file") == 2
    try
        carrier = nrSCSCarrierConfig('SubcarrierSpacing',scs(k), ...
            'NStartGrid',0, 'NSizeGrid',nrb(k));
        bwp = nrWavegenBWPConfig('BandwidthPartID',1, ...
            'SubcarrierSpacing',scs(k), 'NStartBWP',0, 'NSizeBWP',nrb(k));
        pdsch = nrWavegenPDSCHConfig('BandwidthPartID',1, ...
            'Modulation',mo, 'PRBSet',0:nrb(k)-1);
        ssb   = nrWavegenSSBurstConfig('Enable',false);
        pdcch = nrWavegenPDCCHConfig('Enable',false);
        dl = nrDLCarrierConfig('FrequencyRange','FR1', ...
            'ChannelBandwidth',bwM(k), 'NumSubframes',ceil(cfg.poolDuration*1e3), ...
            'SampleRate',cfg.fs, 'SCSCarriers',{carrier}, ...
            'BandwidthParts',{bwp}, 'SSBurst',ssb, 'PDCCH',{pdcch}, ...
            'PDSCH',{pdsch});
        w = nrWaveformGenerator(dl);
        e.x  = double(w(:,1));
        e.bw = nrb(k)*12*scs(k)*1e3;
        e.subtype = sprintf("NR %d kHz SCS, %d MHz, %s", scs(k), bwM(k), mo);
        e.source  = "5G Toolbox nrWaveformGenerator";
    catch err
        warning("p2:nrDefault", ...
            "Custom NR configuration failed (%s). Using the default carrier.", err.message);
        try
            dl = nrDLCarrierConfig('NumSubframes',ceil(cfg.poolDuration*1e3), ...
                'SampleRate',cfg.fs);
            w = nrWaveformGenerator(dl);
            e.x  = double(w(:,1));
            e.bw = dl.SCSCarriers{1}.NSizeGrid*12*dl.SCSCarriers{1}.SubcarrierSpacing*1e3;
            e.subtype = "NR default carrier";
            e.source  = "5G Toolbox nrWaveformGenerator (default config)";
        catch
            [e.x, e.bw] = p2_genOFDMLike(scs(k)*1e3, nrb(k), cfg.fs, cfg.poolDuration);
            e.subtype = sprintf("NR-like OFDM %d kHz, %d MHz", scs(k), bwM(k));
            e.source  = "Fallback OFDM (5G Toolbox not usable)";
        end
    end
else
    [e.x, e.bw] = p2_genOFDMLike(scs(k)*1e3, nrb(k), cfg.fs, cfg.poolDuration);
    e.subtype = sprintf("NR-like OFDM %d kHz, %d MHz", scs(k), bwM(k));
    e.source  = "Fallback OFDM (5G Toolbox not found)";
    warning("p2:noNR", "5G Toolbox not found: using an NR-like OFDM signal.");
end
e.mask = true(size(e.x));
end
