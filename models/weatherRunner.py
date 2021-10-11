import weatherData
import weatherPredict
import weatherGRU_Model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x_train_scaled = []
    y_train_scaled = []
    x_test_scaled = []
    y_test_scaled = []
    num_x_signals = 0
    num_y_signals = 0
    num_train = 0

    x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled,\
    num_x_signals, num_y_signals, num_train, xScaler, yScaler = weatherData.load_data()
    # print(x_train_scaled)

    if False:
        weatherGRU_Model.build_model(x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled,
                                     num_x_signals, num_y_signals, num_train)
    # build_model(x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled, num_x_signals, num_y_signals, num_train)
    # weatherPredict.plot_comparison(start_idx=100000, length=1000, train=True)

    target_names = ['temperature', 'humidity', 'windspeed']
    start_idx = 0
    weatherPredict.plot_comparison(start_idx, target_names,
                                   x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled, xScaler, yScaler,
                                   length=100, train=True)
