import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Masking, Bidirectional, LSTM, Dense, Permute
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from IPython.display import display

# GPU check
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print("GPU device not found. Đảm bảo bạn đã chọn runtime GPU.")
else:
    print(f"Found GPU at: {device_name}")

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

mask_value = -1.0  # giá trị đánh dấu dữ liệu thiếu

# Temporal settings
train_steps  = 335    # độ dài chuỗi đầu vào
test_steps   = 30     # độ dài chuỗi dự đoán
pred_offset  = 70     # khoảng trống giữa đầu vào và đầu ra
stride       = 30     # bước nhảy khi sinh sequence

# Sensor file mapping
base_path = '/content/sensor_data/'
all_sensors = {
    'sensor1': f'{base_path}sensor1.csv',
    'sensor2': f'{base_path}sensor2.csv',
    'sensor3': f'{base_path}sensor3.csv',
    'sensor4': f'{base_path}sensor4.csv'
}

def interpolate(data, mask=mask_value):
    series = pd.Series([np.nan if x==mask else x for x in data])
    return series.interpolate().bfill().ffill().tolist()


def data_normalize(series_list, mask=mask_value):
    normed, scales = [], []
    for s in series_list:
        # pull out *real* values
        valid = [v for v in s if (v != mask and not np.isnan(v))]
        if valid:
            mn, mx = min(valid), max(valid)
        else:
            mn, mx = 0.0, 1.0

        new_s = []
        for v in s:
            if v == mask or np.isnan(v):
                new_s.append(mask)            # keep it as “to be interpolated”
            else:
                # if constant, just output 0.0
                if mx == mn:
                    new_s.append(0.0)
                else:
                    new_s.append((v - mn) / (mx - mn))
        normed.append(new_s)
        scales.append((mn, mx))
    return normed, scales

def create_lag_features(data, lag_steps):
    out = []
    for series in data:
        for lag in lag_steps:
            rolled = np.roll(series, lag)
            rolled[:lag] = series[0]
            out.append(rolled.tolist())
    return data + out


def create_time_features(time_index):
    hours = np.array([t.hour for t in time_index])
    sin_h = np.sin(2*np.pi*hours/24)
    cos_h = np.cos(2*np.pi*hours/24)
    is_weekend = np.array([1 if t.weekday()>=5 else 0 for t in time_index])
    is_work = np.array([1 if (t.weekday()<5 and 9<=t.hour<18) else 0 for t in time_index])
    return [sin_h.tolist(), cos_h.tolist(), is_weekend.tolist(), is_work.tolist()]


def load_and_align_data(sensor_files, variables, mask=mask_value):
    dfs = {}
    for name, fp in sensor_files.items():
        df = pd.read_csv(fp, parse_dates=['datetime'])
        df.set_index('datetime', inplace=True)
        dfs[name] = df

    idx = list(dfs.values())[0].index
    for df in list(dfs.values())[1:]:
        idx = idx.intersection(df.index)

    data, names = [], []
    for name, df in dfs.items():
        sub = df.loc[idx]                        # aligned on the common times
        for var in variables:
            if var in sub.columns:
                data.append(sub[var].tolist())
                names.append(f"{name}_{var}")

    return data, names, idx


def create_sequences(data, n_in, n_out, offset, stride):
    X, Y = [], []
    n_vars = len(data)
    length = len(data[0])
    window = n_in + offset + n_out
    for i in range(0, length-window+1, stride):
        xin = [series[i:i+n_in] for series in data]
        yout = [series[i+n_in+offset:i+n_in+offset+n_out] for series in data]
        X.append(np.stack(xin, axis=1))  # shape (n_in, n_vars)
        Y.append(np.stack(yout, axis=1)) # shape (n_out, n_vars)
    return np.array(X), np.array(Y)

def bilstm_model(n_steps, n_vars, n_out, units=64, lr=1e-4):
    model = Sequential([
        Masking(mask_value=mask_value, input_shape=(n_steps, n_vars)),
        Bidirectional(LSTM(units, return_sequences=False, activation='relu')),
        Dense(n_out * n_vars)
    ])
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr))
    return model


def spatio_temporal_model(n_steps, n_vars, n_out, temp_units=64, spat_units=32, lr=1e-4):
    inp = Input((n_steps, n_vars))
    t = Bidirectional(LSTM(temp_units, return_sequences=True))(inp)
    s = Permute((2,1))(t)
    s = Bidirectional(LSTM(spat_units))(s)
    out = Dense(n_out * n_vars)(s)
    m = Model(inp, out)
    m.compile(loss='mean_absolute_error', optimizer=Adam(lr))
    return m

def Proxy_Learner(data_in, data_out, time_index,
                  train_time, test_time, offset, stride,
                  use_lag=True, use_time=True,
                  lag_steps=[1,3,6],
                  temp_units=64, spat_units=32, lr=1e-4):
    # features
    feats_in = data_in.copy()
    if use_time:
        time_feats = create_time_features(time_index)
        feats_in += time_feats
    if use_lag:
        feats_in = create_lag_features(feats_in, lag_steps)
    # normalize & interp
    norm_in, _ = data_normalize(feats_in)
    norm_out, scales = data_normalize(data_out)
    interp_in = [interpolate(s) for s in norm_in]
    interp_out = [interpolate(s) for s in norm_out]
    # create sequences
    X, Y = create_sequences(interp_in + interp_out, train_time, test_time, offset, stride)
    # split
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=42)
    # choose model
    model = spatio_temporal_model(train_time, X.shape[2], test_time, temp_units, spat_units, lr)
    es = EarlyStopping('val_loss', patience=10, restore_best_weights=True)
    model.fit(X_tr, Y_tr.reshape(len(Y_tr), -1),
              validation_split=0.2, epochs=50, batch_size=16,
              callbacks=[es], verbose=1)
    pred = model.predict(X_te)
    pred = pred.reshape(len(pred), test_time, -1)
    return pred, Y_te, model, scales

def evaluate_predictions(Y_true, Y_pred, names, scales):
    results = {}
    for i, name in enumerate(names):
        true = Y_true[:,:,i].ravel()
        pred = Y_pred[:,:,i].ravel()
        mae = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        r2 = r2_score(true, pred)
        mape = mean_absolute_percentage_error(true+1e-10, pred+1e-10)*100
        results[name] = dict(MAE=mae, RMSE=rmse, R2=r2, MAPE=mape)
        print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, MAPE={mape:.1f}%")
    return results


def plot_correlations(data_in, data_out, names_in, names_out):
    df = pd.DataFrame(np.vstack([data_in, data_out]).T, columns=names_in+names_out)
    corr = df.corr()
    plt.figure(figsize=(10,8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')  # Save instead of show for compatibility
    return corr

if __name__ == '__main__':
    # select variables
    vars_in  = ['CO2.ppm.', 'Humidity..RH.', 'Temperature.C.']
    vars_out = ['PM2_5.ug.m3.']

    # Load input and output data separately
    data_in, names_in, t_idx = load_and_align_data(all_sensors, vars_in)
    data_out, names_out, _ = load_and_align_data(all_sensors, vars_out)

    # correlation analysis
    corr = plot_correlations(data_in, data_out, names_in, names_out)

    # train proxy learner
    pred, true, model, scales = Proxy_Learner(
        data_in, data_out, t_idx,
        train_steps, test_steps, pred_offset, stride,
        use_lag=True, use_time=True,
        lag_steps=[1,3,6,12],
        temp_units=64, spat_units=32, lr=1e-4
    )

    # evaluate
    # Only evaluate the output variables (PM2_5.ug.m3.)
    names = names_out
    evaluate_predictions(true, pred, names, scales)

