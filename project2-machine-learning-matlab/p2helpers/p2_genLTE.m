function e = p2_genLTE(cfg)
%P2_GENLTE  LTE downlink E-UTRA test model waveform (LTE Toolbox).
%
%   Random test model (E-TM 1.1, 1.2, 3.1, 3.2, 3.3) and bandwidth
%   (5, 10, 15, 20 MHz). Without LTE Toolbox an LTE-numerology OFDM signal
%   is generated instead, and e.source says so.

bws  = ["5MHz" "10MHz" "15MHz" "20MHz"];
nrb  = [25 50 75 100];
tmns = ["1.1" "1.2" "3.1" "3.2" "3.3"];
b = randi(numel(bws));
t = randi(numel(tmns));

ok = false;
if exist("lteTestModelTool","file") == 2
    try
        [w, ~, tm] = lteTestModelTool(char(tmns(t)), char(bws(b)));
        [e.x, e.mask] = p2_toCommonRate(w(:,1), true(size(w,1),1), ...
            tm.SamplingRate, cfg.fs);
        e.bw      = tm.NDLRB*180e3;
        e.subtype = "E-TM " + tmns(t) + " " + bws(b);
        e.source  = "LTE Toolbox lteTestModelTool";
        ok = true;
    catch
        ok = false;
    end
end
if ~ok
    [e.x, e.bw] = p2_genOFDMLike(15e3, nrb(b), cfg.fs, cfg.poolDuration);
    e.mask    = true(size(e.x));
    e.subtype = "LTE-like OFDM " + bws(b);
    e.source  = "Fallback OFDM (LTE Toolbox not found)";
    warning("p2:noLTE", "LTE Toolbox not available: using an LTE-like OFDM signal.");
end
end
