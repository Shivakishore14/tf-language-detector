import config
import os
import numpy as np


MAX_LEN_X = 0
NUM_LINES = 10
MAX_SEQ_LENGTH = 20


def _preprocess_data(content):
    return content.replace(' ', ' <!space!> '). \
        replace('\n', ' <!new-line!> '). \
        replace('\t', ' <!tab!> ')


def process_lines(lines):
    n_lines = len(lines)
    x_data_raw = None
    if n_lines / NUM_LINES >= 1:
        extra = n_lines % NUM_LINES
        lines = [line[: MAX_SEQ_LENGTH] for line in lines]
        per_x_lines = np.reshape(lines[: -extra], (-1, NUM_LINES))
        x_data_raw = ['\n'.join(line_set) for line_set in per_x_lines]
    else:
        x_data_raw = lines
    x_data = [_preprocess_data(data) for data in x_data_raw if data != '']
    return x_data


def read_file(filename):
    print filename
    with open(filename, 'r') as f:
        lines = f.readlines()
        return process_lines(lines)


def get_x_y_train_test():
    global MAX_LEN_X

    def find_max(x):
        max_len = max([len(row) for row in x])
        print("max len {}".format(max_len))
        return max_len

    def read_dataset_and_generate_x_y():
        x = []
        y = []
        for dataset_path in config.dataset_paths:
            class_ = dataset_path.split('/')[-1]
            files_ = [os.path.join(dataset_path, file_) for file_ in os.listdir(dataset_path)]
            for file_ in files_:
                data_x = read_file(file_)
                len_data_x = len(data_x)

                x.extend(data_x)
                data_y = np.eye(len(config.classes))[config.classes.index(class_)]
                y_encoded = [data_y for _ in range(len_data_x)]
                # y_encoded = config.classes.index(class_)
                y.extend(y_encoded)
        return x, y

    x, y = read_dataset_and_generate_x_y()

    MAX_LEN_X = find_max(x)

    n_rows = len(y)
    shuffle_indices = np.random.permutation(np.arange(n_rows))
    x_test = [x[i] for i in shuffle_indices[:n_rows/9]]
    y_test = [y[i] for i in shuffle_indices[:n_rows/9]]

    x_train = [x[i] for i in shuffle_indices[n_rows/9:]]
    y_train = [y[i] for i in shuffle_indices[n_rows/9:]]

    return x_train, y_train, x_test, y_test


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_max_len():
    get_x_y_train_test()
    return MAX_LEN_X
