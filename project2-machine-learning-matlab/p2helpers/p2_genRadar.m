function e = p2_genRadar(cfg)
%P2_GENRADAR  Pulsed radar: linear-FM chirp or Barker-13 coded pulses.
%
%   Pulse width 5-40 us, duty cycle 5-25 %, small PRI jitter.
%   LFM sweep 1-20 MHz (up or down). Barker chip rate = 13 / pulse width.

fs    = cfg.fs;
total = round(cfg.poolDuration*fs);
tau   = 5e-6 + rand*35e-6;
duty  = 0.05 + rand*0.20;
pri   = tau/duty;
nP    = round(tau*fs);
t     = (0:nP-1)'/fs;

if rand < 0.7
    B = 1e6 + rand*19e6;
    dirn = sign(rand - 0.5);
    pulse = exp(1j*dirn*pi*(B/tau)*(t - tau/2).^2);
    e.bw = B;
    e.subtype = sprintf("LFM %.1f MHz sweep, %.0f us", B/1e6, tau*1e6);
else
    barker = [1 1 1 1 1 -1 -1 1 1 -1 1 -1 1];
    chipIdx = min(13, floor(t/(tau/13)) + 1);
    pulse = complex(barker(chipIdx).');
    e.bw = 13/tau;
    e.subtype = sprintf("Barker-13, %.0f us", tau*1e6);
end

x = zeros(total,1);
mask = false(total,1);
pos = 1 + round(rand*pri*fs);
while pos + nP - 1 <= total
    x(pos:pos+nP-1) = pulse;
    mask(pos:pos+nP-1) = true;
    pos = pos + round(pri*fs*(1 + 0.02*randn));
end
e.x = x;
e.mask = mask;
e.source = "Own pulsed-radar generator (p2_genRadar)";
end
