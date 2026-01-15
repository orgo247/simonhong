import datetime as dt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Importing historical BTC price data (2019–present).
def load_data(start_date, end_date):
    sym = "BTC-USD"
    df = yf.download(sym, start=start_date, end=end_date)
    df.index = pd.to_datetime(df.index,errors="coerce")
    df = df.sort_index()
    #Selecting the daily high price column.
    df = df[["High"]]

    if df.empty:
        raise ValueError(f"No data for Bitcoin between '{start_date}' '{end_date}'.")
    return df


#Building an LSTM neural network with Keras.
#Training and evaluating using RMSE or MAE.
#Visualizing predicted vs actual prices on the test set.

# Visualize data
def visualize_data(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df["High"], label="Bitcoin High Price")
    plt.title("Bitcoin High Price 2019-2025")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.show()
# Split data into train and test sets
def test_train_split(df):
    train, test = df.loc["2019-01-01":"2022-12-31"].copy(), df.loc["2023-01-01":"2025-12-08"].copy()
    return train, test
# Scaling and windowing data in 30 day input -> next day prediction
def preprocess_data(df, scaler, time_step=30, fit_scaler=True):
    data = df["High"].values
    if fit_scaler:
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)
    X,y = [],[]
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    return np.reshape(X, (X.shape[0], X.shape[1], 1)), y

# Build LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error')
    return model


def main():
    # set date
    start_date = "2019-01-01"
    end_date = "2025-12-08"
    # load date given date
    df = load_data(start_date, end_date)
    # visualize data
    visualize_data(df)
    # split data into train and test
    train, test = test_train_split(df)
    # specify scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train, y_train = preprocess_data(train, scaler, time_step=30, fit_scaler=True)
    X_test, y_test, = preprocess_data(test, scaler, time_step=30, fit_scaler=False)
    # build model
    model = build_model(X_train.shape[1:])
    history = model.fit(X_train, y_train,epochs=30, batch_size=10,validation_split=0.1, shuffle=False, verbose=1)

    # plot loss over epochs
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title('Training loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

    #predict
    y_pred = model.predict(X_test).reshape(-1,1)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # RMSE
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse}")

    #plot prediction
    plt.figure(figsize=(12,6))
    test_dates = test.index[30:]
    plt.plot(test_dates, y_test, label="Actual Price", color="blue")
    plt.plot(test_dates, y_pred, label="Predicted Price", color="red")
    plt.title("Bitcoin High Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()


