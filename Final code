# final code to run after loading all the necessary vivado files like .bit, .tcl, .hwh and all should be in the same name 



# Jupyter Setup 
%matplotlib notebook 
import matplotlib.pyplot as plt
import numpy as np
from pynq import Overlay, allocate
from scipy.io import wavfile
import time
from scipy.io.wavfile import write as wavwrite
from pynq.overlays.base import BaseOverlay

base = BaseOverlay("base.bit")
# ─────────────────────────────────────────────
sw0 = base.switches[0].read()
sw1 = base.switches[1].read()
# 1. Plot functions
def plot_to_notebook(time_sec, in_signal, n_samples, out_signal=None): 
    plt.figure()
    plt.subplot(1,1,1)
    plt.xlabel('Time (μs)')
    plt.grid()
    plt.plot(time_sec[:n_samples]*1e6, in_signal[:n_samples], 'y-', label='Input signal')
    if out_signal is not None: 
        plt.plot(time_sec[:n_samples]*1e6, out_signal[:n_samples], 'g-', linewidth=2, label='FIR output') 
    plt.legend()

def plot_frequency_analysis(input_signal, output_signal, fs, n_fft=4096):
    min_len = min(len(input_signal), len(output_signal))
    input_signal = input_signal[:min_len]
    output_signal = output_signal[:min_len]

    freq = np.fft.rfftfreq(n_fft, 1/fs)
    input_fft = np.fft.rfft(input_signal, n=n_fft)
    output_fft = np.fft.rfft(output_signal, n=n_fft)

    input_magnitude = 20 * np.log10(np.abs(input_fft) + 1e-12)
    output_magnitude = 20 * np.log10(np.abs(output_fft) + 1e-12)

    plt.figure(figsize=(10, 5))
    plt.plot(freq, input_magnitude, label="Input Signal", color="orange")
    plt.plot(freq, output_magnitude, label="Filtered Output", color="green")
    plt.title("Frequency Domain Analysis (FFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
# 2. Load FIR overlay
print("Loading overlay...")
overlay = None
if sw0 == 0 and sw1 ==0 : 
    overlay = Overlay('FIR_filter_low1.bit')
elif sw0 == 0 and sw1 == 1:
    overlay = Overlay('FIR_filter_high1.bit')
elif sw0==1 and sw1 == 0:
    overlay = Overlay('FIR_Bandpass_Filter.bit')
else:
    overlay = Overlay('FIR_filter_bandStop1.bit')
    
dma = overlay.filter.fir_dma

# ─────────────────────────────────────────────
# 3. Load recorded audio from file
print("Loading audio...")
fs, samples = wavfile.read("recording_1.wav")

# Convert to mono if stereo
if samples.ndim > 1:
    samples = samples[:, 0]

# Normalize and convert to int32
samples = samples.astype(np.float32)
samples = samples / np.max(np.abs(samples))
samples = (samples * (2**31 - 1)).astype(np.int32)

# Use only first 32768 samples to avoid overloading
samples = samples[:1048576] 
n = len(samples)
t = np.arange(n) / fs

# ─────────────────────────────────────────────
# 4. Allocate buffers and run through FPGA
print("Allocating buffers...")
in_buffer = allocate(shape=(n,), dtype=np.int32)
out_buffer = allocate(shape=(n,), dtype=np.int32)

np.copyto(in_buffer, samples)
in_buffer.flush()
out_buffer.invalidate()

print("Running FIR filter on FPGA...")
start_time = time.time()
try:
    dma.sendchannel.transfer(in_buffer)
    dma.recvchannel.transfer(out_buffer)

    dma.sendchannel.wait()
    dma.recvchannel.wait()

    stop_time = time.time()
    hw_exec_time = stop_time - start_time
    hwAC = (exec_time_comp / hw_exec_time)*100
    print("Execution time:", hw_exec_time)
    print("hardware accelaration percent is " , hwAC)
    filtered_output = np.array(out_buffer)
    filtered_output = np.clip(filtered_output / (2**16), -32768, 32767).astype(np.int16)

    output_filename = "filtered_output.wav"
    wavwrite(output_filename, fs, filtered_output)
    print(f"Filtered audio saved to {output_filename}")

    # ─────────────────────────────────────────────
    # 5. Plot time and frequency domain results
    plot_to_notebook(t, samples, 1000, out_signal=out_buffer)
    plot_frequency_analysis(samples, out_buffer, fs)

except Exception as e:
    print("DMA Transfer failed:", e)

finally:
    # ─────────────────────────────────────────────
    # 6. Cleanup 
    print("Freeing buffers...")
    in_buffer.freebuffer()
    out_buffer.freebuffer()
