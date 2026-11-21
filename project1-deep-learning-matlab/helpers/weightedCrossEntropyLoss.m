function loss = weightedCrossEntropyLoss(ypred, yactual, weights)
%WEIGHTEDCROSSENTROPYLOSS  Class-weighted cross-entropy for segmentation.
%
%   loss = weightedCrossEntropyLoss(ypred, yactual, weights)
%
%   This is the loss function of the MathWorks "Spectrum Sensing with Deep
%   Learning to Identify 5G, LTE, and WLAN Signals" example, reproduced so
%   that training follows the example. Each class is weighted so that the
%   rare classes (Bluetooth bursts are tiny compared with the noise
%   background) count as much as the common one, and the loss is divided
%   by the total weight of the pixels in the mini-batch.

cdim = find(dims(ypred) == 'C');
loss = crossentropy(ypred, yactual, weights, ...
    WeightsFormat="C", NormalizationFactor="none");
wn   = shiftdim(weights(:)', -(cdim-2));
wnT  = extractdata(yactual).*wn;
normFac = sum(wnT(:)) + eps('single');
loss = loss/normFac;
end
