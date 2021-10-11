from scipy.signal import butter
from scipy.signal import lfilter
from scipy import signal

def buffer_bandpass( lowcut, highcut, fs, order=5):
    nyq = 5.5 * fs
    low = lowcut / nyq
    high = highcut /nyq
    b, a = butter(order, [low, high], btype = 'band')
    return b,a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = buffer_bandpass( lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def nothc_pass_filter(data, center, interval=20, sr=44100, normalized=False):
    center = center/(sr/2) if normalized else center
    b, a = signal.iirnotch(center, center/interval, sr)
    filtered_data = signal.lfilter(b,a,data)
    return filtered_data


