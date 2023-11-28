---
title: "COLA"
date: 2023-11-28
categories:
  - "Speech Enhancement"
tags:
  - "STFT"
  - "COLA"
sidebar: false
---

## Constant Overlap-Add Constraint

To ensure successful reconstruction of nonmodified spectra, the analysis window must satisfy the COLA constraint. In general, if the analysis window satisfies the condition $\displaystyle \sum\_{m=-\infty}^\infty g^{a+1}(n-mR)=c,\\;^\forall n\in\mathbb{Z}$, the window is considered to be COLA-compliant.

## Perfect Reconstruction

In general, computing the STFT of an input signal and inverting it does not result in perfect reconstruction. If you want the output of ISTFT to match the original input signal as closely as possible, the signal and the window must satisfy the following conditions:
- Input size - If you invert the output of stft using istft and want the result to be the same length as the input signal $x$, the value of $k=\dfrac{N\_x-L}{M-L}$ must be an integer. In the equation, $N\_x$ is the length of the signal, $M$ is the length of the window, and $L$ is the overlap length.
- COLA compliance - Use COLA-compliant windows, assuming that you have not modified the short-time Fourier transform of the signal.
- Padding - If the length of the input signal is such that the value of $k$ is not an integer, zero-pad the signal before computing the short-time Fourier transform. Remove the extra zeros after inverting the signal.

## Python Code for Example

```py
import scipy as sp
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
```

### COLA-non-compliant Window

```py
HOP_LEN = 160
WIN_LEN = 512
NUM_WINDOWS = 30
```

```py
window = sp.signal.windows.hann(WIN_LEN, sym=False)
sp.signal.check_COLA(window, WIN_LEN, WIN_LEN - HOP_LEN) # False
```

```py
window_add = np.zeros(HOP_LEN * NUM_WINDOWS + WIN_LEN - HOP_LEN)

fig, ax = plt.subplots(figsize=(12, 3.5))
for i in range(NUM_WINDOWS):
    ax.plot(np.arange(HOP_LEN * i, HOP_LEN * i + WIN_LEN), window, c='gray')
    for j in range(WIN_LEN):
        window_add[HOP_LEN * i + j] += window[j]
ax.plot(window_add, c='black')
plt.show()
```
{{<figure src="/speech_enhancement/cola1.png" width="800">}}

### COLA-compliant Window

```py
HOP_LEN = 160
WIN_LEN = 480
NUM_WINDOWS = 30
```

```py
window = sp.signal.windows.hann(WIN_LEN, sym=False)
sp.signal.check_COLA(window, WIN_LEN, WIN_LEN - HOP_LEN) # True
```

```py
window_add = np.zeros(HOP_LEN * NUM_WINDOWS + WIN_LEN - HOP_LEN)

fig, ax = plt.subplots(figsize=(12, 3.5))
for i in range(NUM_WINDOWS):
    ax.plot(np.arange(HOP_LEN * i, HOP_LEN * i + WIN_LEN), window, c='gray')
    for j in range(WIN_LEN):
        window_add[HOP_LEN * i + j] += window[j]
ax.plot(window_add, c='black')
plt.show()
```
{{<figure src="/speech_enhancement/cola2.png" width="800">}}

### Application

```py
HOP_LEN = 160
WIN_LEN = 480
NUM_WINDOWS = 160000 // HOP_LEN
```

```py
signal, sr = sf.read("clnsp113_car_19980_0_snr18_tl-30_fileid_178.wav")
```

```py
signal_add = np.zeros(len(signal))

for i in range(NUM_WINDOWS):
    for j in range(WIN_LEN):
        if HOP_LEN * i + j >= len(signal):
            break
        signal_add[HOP_LEN * i + j] += signal[HOP_LEN * i + j] * window[j]
```

```py
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
ax[0].plot(signal, linewidth=0.1)
ax[0].set_title("Original")
ax[1].plot(signal_add / 1.5, linewidth=0.1)
ax[1].set_title("Reconstructed")
plt.show()
```
{{<figure src="/speech_enhancement/cola3.png" width="800">}}

---

**Reference**

1. https://www.mathworks.com/help/signal/ref/stft.html
2. https://www.mathworks.com/help/signal/ref/istft.html
3. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_COLA.html
