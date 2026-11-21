function e = p2_genPoolEntry(cls, cfg)
%P2_GENPOOLENTRY  One random base waveform of class cls at cfg.fs.
%
%   e = p2_genPoolEntry(cls, cfg) returns a struct with fields
%     x        complex single column, baseband, cfg.poolDuration long
%     mask     true where the signal is on (packets, pulses)
%     bw       occupied bandwidth used to define SNR (Hz)
%     subtype  what was generated, e.g. "802.11n HT MCS 4"
%     source   which MATLAB function produced it

switch string(cls)
    case "WLAN",      e = p2_genWLAN(cfg);
    case "Bluetooth", e = p2_genBluetooth(cfg);
    case "ZigBee",    e = p2_genZigBee(cfg);
    case "LTE",       e = p2_genLTE(cfg);
    case "NR5G",      e = p2_genNR(cfg);
    case "Radar",     e = p2_genRadar(cfg);
    otherwise
        error("p2:unknownClass", "No generator for class %s.", cls);
end

% Fix the length to cfg.poolDuration (repeat if too short).
n = round(cfg.poolDuration*cfg.fs);
if numel(e.x) < n
    r = ceil(n/numel(e.x));
    e.x = repmat(e.x, r, 1);
    e.mask = repmat(e.mask, r, 1);
end
e.x    = single(e.x(1:n));
e.mask = e.mask(1:n);
e.cls  = string(cls);
end
