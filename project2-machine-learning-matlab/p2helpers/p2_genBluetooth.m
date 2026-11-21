function e = p2_genBluetooth(cfg)
%P2_GENBLUETOOTH  Bluetooth BR/EDR packets (Bluetooth Toolbox).
%
%   The packets are kept at baseband. Frequency hopping across the band
%   is produced later by the random frequency offset of levels 2 to 6.

sps   = 8;                     % 1 Msym/s -> 8 MHz
fsB   = sps*1e6;
total = round(cfg.poolDuration*fsB);

modes      = ["BR" "BR" "EDR2M" "EDR3M"];
types      = ["DH1" "DH3" "DH5"];
maxPayload = [27 183 339; 54 367 679; 83 552 1021];
phyMode = modes(randi(numel(modes)));

x = zeros(0,1);
mask = false(0,1);
while numel(x) < total
    gap = round((100e-6 + rand*1.2e-3)*fsB);
    ti = randi(3);
    mx = maxPayload(find(phyMode == ["BR" "EDR2M" "EDR3M"],1), ti);
    try
        c = bluetoothWaveformConfig('Mode',char(phyMode), ...
            'PacketType',char(types(ti)), 'SamplesPerSymbol',sps);
        c.PayloadLength = randi([ceil(0.3*mx) mx]);
        w = bluetoothWaveformGenerator(randi([0 1],getPayloadLength(c)*8,1), c);
        sub = phyMode + " " + types(ti);
    catch err
        warning("p2:btFallback", "Bluetooth %s/%s failed (%s). Using BR.", ...
            phyMode, types(ti), err.message);
        c = bluetoothWaveformConfig('Mode','BR', ...
            'PacketType',char(types(ti)), 'SamplesPerSymbol',sps);
        c.PayloadLength = maxPayload(1,ti);
        w = bluetoothWaveformGenerator(randi([0 1],getPayloadLength(c)*8,1), c);
        sub = "BR " + types(ti);
    end
    w = double(w(:));
    last = find(abs(w) > 0, 1, "last");          % the generator pads zeros
    w = w(1:last);
    x    = [x; zeros(gap,1); w];                     %#ok<AGROW>
    mask = [mask; false(gap,1); true(numel(w),1)];   %#ok<AGROW>
end

[e.x, e.mask] = p2_toCommonRate(x, mask, fsB, cfg.fs);
e.bw      = 1e6;
e.subtype = sub;
e.source  = "Bluetooth Toolbox bluetoothWaveformGenerator";
end
