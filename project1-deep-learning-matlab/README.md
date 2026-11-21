# Project 1: Deep-learning classification of interfering RF signals (Bluetooth and WiFi)

A MATLAB project that follows the handout's suggested steps one for one. It uses transfer learning to adapt the network from MathWorks' *Spectrum Sensing with Deep Learning to Identify 5G and LTE Signals* example so that it finds **Bluetooth** and **WLAN (WiFi)** signals in a wideband capture of the 2.4 GHz band.

**Status: the code has not been run.** It was written without access to MATLAB or an SDR. Section 8 lists exactly what was checked and what was not. Run it on your machine before relying on it, and expect to fix small things.

---

## 1. Handout steps and where they are done

| Handout step | File | What you get |
|---|---|---|
| 1. Generate two signals with the Wireless Waveform Generator app | `step1_generate_waveforms.m` (or the app, section 5) | `waveforms/bluetoothWaveform.mat`, `waveforms/wlanWaveform.mat` |
| 2. Frequency-translate with the Multiband Combiner so they do not overlap | `step2_frequency_translate.m` | `waveforms/combinedWaveform.mat`, before/after spectrum plots |
| 3.1 Create a training and validation data set with a variety of impairments | `step3a_create_dataset.m` | `data/train`, `data/val`, `data/test` (images, labels, `meta.csv`) |
| 3.2 Train by transfer learning from the example's network | `step3b_train_network.m` | `model/rfClassifierNet.mat` |
| 3.3 Validate with a small percentage of the generated data | `step3c_validate_network.m` | accuracy, IoU, confusion matrix, accuracy against SNR |
| 3.4 Test with over-the-air signals from an SDR | `step4_test_over_the_air.m` | labelled captures in `results/ota` |
| Advanced: signals that overlap in frequency | `advanced_overlap_test.m` | metrics for overlapping Bluetooth and WLAN |

All settings live in `projectConfig.m`. Supporting functions are in `helpers/`.

## 2. What you need

- **MATLAB R2024a or newer.** The code uses `deeplabv3plus` and `trainnet`, which older releases do not have.
- **Toolboxes:** Communications, DSP System, Signal Processing, WLAN, Bluetooth, Deep Learning, Computer Vision, Image Processing.
- **Add-on:** *Deep Learning Toolbox Model for ResNet-18 Network* (free, from the Add-On Explorer).
- **Optional:** Parallel Computing Toolbox (faster data generation, GPU training). A GPU makes training practical; on a CPU it will be slow.
- **For step 3.4:** an ADALM-PLUTO with *Communications Toolbox Support Package for Analog Devices ADALM-PLUTO Radio*. Other radios are covered in section 7.

## 3. Getting the example network (needed for step 3.2)

The handout says to use the existing network in the example. That network is a file called `specSenseTrainedNetResnet18.mat`, which the example downloads.

1. In MATLAB run `openExample('5g/SpectrumSensingWithDeepLearning5GLTEExample')`. The example lists 5G Toolbox and LTE Toolbox among its requirements. If MATLAB cannot open or run it because they are missing, use the fallback described below.
2. Run the example once with its default settings. It downloads the trained networks.
3. Find the file with `which specSenseTrainedNetResnet18.mat` and copy it into this project's `pretrained` folder.

If the file is missing, `step3b_train_network` still runs: it starts from the ImageNet-pretrained ResNet-18 backbone only, prints a warning, and says so in its output. That is still transfer learning (it is what the example itself does), but it is not the handout's wording, so get the file if you can.

## 4. How to run

**Easiest:** open `RUN_ME.m` and press Run. It checks your toolboxes, fetches the example network, picks run settings for your computer (GPU or not), runs every step, and packs all results into `Project1_results_to_send.zip`. See `HOW_TO_RUN.txt`.

You can also run the step scripts one at a time in the order of the table above, or type `runAll`. Each step reads the files the previous one saved.

## 5. Step 1 in the app

The handout asks for the Wireless Waveform Generator app. `step1_generate_waveforms.m` produces the same two waveforms in code, using the functions the app itself calls, so the project runs without any clicking. To do it in the app as written:

1. Run `wirelessWaveformGenerator`.
2. **Bluetooth.** In the Waveform Type section choose Bluetooth BR/EDR. Set mode BR, packet type DH5, samples per symbol 20 (this gives a 20 MHz sample rate). Press **Generate**.
3. Export the waveform (Export, to a MAT-file or to the workspace) and save it as `waveforms/bluetooth_app.mat`.
4. **WLAN.** Choose a WLAN waveform (802.11n/ac, HT format), 20 MHz bandwidth, MCS 4, PSDU length 2000 bytes, 8 packets, idle time 0.5 ms. Press **Generate**, export, and save as `waveforms/wlan_app.mat`.
5. Run `step1_generate_waveforms`. It finds the two `_app.mat` files and uses them instead of generating its own.

Menu names differ a little between releases. The loader looks inside the file for the first complex vector and for a value called `Fs` or `SampleRate`; if no sample rate is stored it assumes 20 MHz and warns you.

## 6. How it works

**The view of the band.** Every frame is 20 ms of signal, 60 MHz wide, centred on 2.44 GHz. That covers most of the 2.4 GHz band, where WiFi and Bluetooth share spectrum. 60 MHz is used rather than the example's 61.44 MHz because it is a whole multiple of the native rates of both signals (20 MHz and 4 MHz), so the Multiband Combiner interpolates by whole numbers.

**The image.** `rfSpectrogramImage` turns a frame into a 256 x 256 colour image: frequency left to right, time top to bottom. The same function is used for training data and for SDR captures. The colour scale is tied to each frame's noise floor, so the picture does not depend on receiver gain.

**The classes.** Each pixel is Noise, Bluetooth or WLAN. This is semantic segmentation, the same method as the example: the network says *where* in time and frequency each signal is, not only which signals are present.

**The labels.** Unlike the example's signals, WiFi and Bluetooth are bursty: WiFi sends packets with gaps, and Bluetooth hops to a new 1 MHz channel for each packet. So labels are drawn per packet, from the known start time, end time and frequency of each one. Nothing is labelled by hand.

**The signals in the data set.**

- Bluetooth BR and EDR data packets of 1, 3 and 5 slots, hopping over 2 to 6 channels per frame.
- WLAN on one 20 MHz channel per frame: a mix of 802.11a/g OFDM, 802.11n HT and 802.11b DSSS/CCK packets (2.4 GHz access points commonly send beacons as DSSS).
- Frame types: 25 % Bluetooth only, 25 % WLAN only, 45 % both in separate parts of the band, 5 % noise only.

**The impairments** (handout: "a variety of impairments"), drawn at random for every frame:

| Impairment | Range | Where |
|---|---|---|
| Noise | SNR 5 to 40 dB, separately per signal | `applyRxImpairments` |
| Multipath fading | TGn models A to E (WLAN); Rician gain per hop (Bluetooth) | `generateLabeledFrame`, `genBluetoothFrame` |
| Transmitter frequency error | up to 50 kHz (WLAN), 75 kHz (Bluetooth) | `generateLabeledFrame` |
| Phase noise | receiver oscillator, 25 % of frames | `applyRxImpairments` |
| IQ imbalance | up to 0.2 dB and 1.5 degrees, 70 % of frames | `applyRxImpairments` |
| DC offset (LO leakage) | spike at band centre, 70 % of frames | `applyRxImpairments` |
| Timing | random start, packet lengths, gaps, hop pattern | `genWlanFrame`, `genBluetoothFrame` |

**Transfer learning.** `buildTransferNetwork` builds the example's architecture (DeepLab v3+ on ResNet-18) with an output for the three classes here, then copies in every weight from the example's trained network whose layer name and size match. Only the final class-scoring layers differ, so only they start from scratch, and they are given a higher learning rate. The script prints how many weight arrays were copied, so you can see that the transfer happened. The loss function and class weighting are taken from the example.

## 7. Over-the-air test (step 3.4)

`step4_test_over_the_air.m` uses, in this order: a connected PLUTO, recordings in `captures/`, or one simulated frame. The simulated frame only shows that the processing runs. It is labelled as simulated on screen and is not an over-the-air result; do not report it as one.

**Receiving existing signals.** With a PLUTO connected the script captures five 20 ms frames around 2.44 GHz. Nearby WiFi routers and Bluetooth devices (stream audio to a Bluetooth speaker next to the antenna for a strong test signal) should appear. There is no ground truth for these captures, so judge them by eye against what you know is transmitting.

**Transmitting your own waveform.** Set `cfg.ota.transmit = true` to radiate the step-2 waveform from the same PLUTO while it receives. Connect TX to RX with a cable and a 30 dB attenuator, or keep the gain low with antennas. Radio transmission is regulated; check what your lab and your country allow in this band before transmitting.

**Sample rate.** The script asks the PLUTO for 60 MHz in burst mode, following the MathWorks example, which uses 61.44 MHz. If you see overflow warnings, the captures are not contiguous and should not be used.

**Other radios.** Any recording saved as `captures/name.mat` with variables `iq` (complex samples) and `fs` is processed. Recordings above 60 MHz are resampled; recordings below 60 MHz are skipped, because the network was trained on 60 MHz-wide frames. For a radio with less bandwidth, lower `cfg.fs` (it must be a multiple of 20 MHz, for example 40e6, with `cfg.wlanMaxOffset = 10e6` and `cfg.btMaxOffset = 18e6`), then regenerate the data and retrain.

## 8. What was and was not checked

Checked:

- Every toolbox function call was compared with the current MathWorks documentation (R2026b pages): `bluetoothWaveformGenerator`, `bluetoothWaveformConfig`, `wlanWaveformGenerator`, `wlanNonHTConfig`, `wlanTGnChannel`, `comm.MultibandCombiner`, `deeplabv3plus`, `trainnet`, `semanticseg`, `dlnetwork`, and the PLUTO receiver and transmitter objects.
- All 22 files pass a MATLAB syntax parser.
- The spectrogram and label geometry (`rfSpectrogramImage`, `boxesToLabelImage`) was ported to Python and tested with synthetic bursts at known times and frequencies. Label rows and columns matched the expected positions exactly, and every labelled pixel contained signal energy.

Not checked:

- Nothing was executed in MATLAB. Data generation, training, validation and the SDR step are untested.
- No accuracy figures exist. Any number you report must come from your own run of `step3c_validate_network`.
- Run times are unknown.

Places most likely to need a small fix:

- **Bluetooth EDR packet names** differ between releases. If your release rejects them, `genBluetoothFrame` falls back to basic-rate packets and prints a warning; the data set is then BR only.
- **GPU memory.** If training stops with an out-of-memory error, lower `cfg.miniBatchSize` to 8.
- **PLUTO at 60 MHz**, as described in section 7.

## 9. Limits to keep in mind when you interpret results

- A Bluetooth burst is small in the image, about 5 pixels wide and 5 to 40 pixels long. Expect its pixel-level scores to be lower than WLAN's, mostly from errors at the edges of bursts.
- Bluetooth Low Energy, ZigBee, microwave ovens and other 2.4 GHz users are not in the training set. Real captures contain them, and the network will label them as whichever class they most resemble.
- Only 20 MHz WLAN channels are generated. 40 MHz channels are not.
- Simulated DSSS uses a plain rectangular chip pulse, so its spectrum shape is approximate.

## 10. Advanced work in the handout

- **Overlapping signals:** done in `advanced_overlap_test.m`, with instructions inside for retraining with overlap.
- **Editing the network in Deep Network Designer:** after `[net0, report] = buildTransferNetwork(projectConfig)`, run `deepNetworkDesigner(net0)`.
- **Other signal pairs:** add a generator like `genBluetoothFrame.m` for the new signal, add its name to `cfg.classNames` and `cfg.labelIDs`, and call it from `generateLabeledFrame.m`.
- Not attempted: pruning and quantisation, public data sets, C/C++ and HDL code generation.

## 11. Background

- MathWorks example: *Spectrum Sensing with Deep Learning to Identify 5G, LTE, and WLAN Signals* (Communications Toolbox documentation).
- `comm.MultibandCombiner`, `bluetoothWaveformGenerator`, `wlanWaveformGenerator`, `deeplabv3plus` reference pages.
