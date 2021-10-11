import os
import sys
import libs.waveletDecomposer as waveDec

sys.path.append(os.path.dirname(os.path.abspath((os.path.dirname(__file__)))))

# waveDec.show_fft_cases()

# waveDec.show_wavelet_case()

# waveDec.show_coefficient()

# waveDec.show_loadedData()

uci_har_signals_train = []
uci_har_labels_train = []
uci_har_signals_test = []
uci_har_labels_test = []


#uci_har_signals_train,\
#uci_har_labels_train,\
#uci_har_signals_test,\
#uci_har_labels_test = waveDec.get_ucihar_data()

waveDec.execute_load_uci()