def window(data, label, size, stride):
    x, y = [], []
    for i in range(0, len(label), stride):
        if i+size < len(label):
            l = set(label[i:i+size])
            if len(l) > 1 or label[i] == 0:
                continue
            elif len(l) == 1:
                x.append(data[i: i + size, :])
                y.append(label[i])
    return x, y

def split(result, test_size):
    x_train, x_test, y_train, y_test = [], [], [], []
    for i, data in enumerate(result):
        label = [i for n in range(len(data))]
        x_train_, x_test_, y_train_, y_test_ = train_test_split(data, label, test_size=test_size, shuffle=True)
        x_train.extend(x_train_)
        y_train.extend(y_train_)
        x_test.extend(x_test_)
        y_test.extend(y_test_)
    return x_train, y_train, x_test, y_test

def sliding_window(time_series, width, step, order='F'):
    w = np.hstack([time_series[i:1 + i - width or None:step] for i in range(0, width)])
    result = w.reshape((int(len(w) / width), width), order='F')
    if order == 'F':
        return result
    else:
        return np.ascontiguousarray(result)
