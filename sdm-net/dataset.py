import copy
import json

import numpy as np
import sklearn.preprocessing as preprocessing
from scipy.stats import multivariate_normal
import torch
import random as rand
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import net


# Fourier feature mapping
def input_basis_mapping(x_ip, B):
    # https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb
    x = np.zeros((x_ip.shape[0], x_ip.shape[1]))
    # want `wrapped` encoding
    # x_ip[:,0] is min -180, max 180
    # x_ip[:,1] is min -90, max 90
    x[:, 0] = x_ip[:, 0]
    x[:, 1] = x_ip[:, 1]
    x_proj = (np.pi*x) @ B.T
    return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1).astype(np.float32)

# Preprocessing function - performs input mapping, then transforms to a tensor
def convert_loc_to_tensor(x, B=None, mapping=None, device=None):
    # intput is in lon {-180, 180}, lat {90, -90}
    xt = x.astype(np.float32)
    xt[:,0] /= 180.0
    xt[:,1] /= 90.0
    if mapping is not None:
        xt = input_basis_mapping(xt, B)
    xt = torch.from_numpy(xt)
    if device is not None:
        xt = xt.to(device)
    return xt

# For synthetic data generation - manual wrap-around for when a gaussian generates points outside of the grid
def wrap_around(train, test):
    for i in range(0, len(train)):
        if train[i][0] < -180:
            train[i][0] = 180 + (train[i][0] + 180)
        if train[i][0] > 180:
            train[i][0] = -180 + (train[i][0] - 180)
        if train[i][1] < -90:
            train[i][1] = 90 + (train[i][1] + 90)
        if train[i][1] > 90:
            train[i][1] = -90 + (train[i][1] - 90)

    for i in range(0, len(test)):
        if test[i][0] < -180:
            test[i][0] = 180 + (test[i][0] + 180)
        if test[i][0] > 180:
            test[i][0] = -180 + (test[i][0] - 180)
        if test[i][1] < -90:
            test[i][0] = 90 + (test[i][1] + 90)
        if test[i][1] > 90:
            test[i][0] = -90 + (test[i][1] - 90)

    return train, test

# Writes the generated dataset to csv files
def write_to_files(train, test, absence, dir):
    train.to_csv(dir + '/train.csv')
    test.to_csv(dir + '/val.csv')
    absence.to_csv(dir + '/absences.csv')
    print("Written csv.")

# Code for generating equal classes
def create_presences_equal(NUM_CLASS):
    x_train = np.empty((1, 2))
    y_train = np.empty(1)
    x_val = np.empty((1, 2))
    y_val = np.empty(1)

    for i in range(0, NUM_CLASS):
        print("Generating Gaussian for species", i)
        num_gauss = np.random.randint(10)
        gauss_distribution = []

        if (num_gauss < 4):
            num_gauss = 1
        elif (num_gauss < 7):
            num_gauss = 2
        elif (num_gauss < 9):
            num_gauss = 3
        else:
            num_gauss = 4

        # Generate exactly 50 data points
        k = 0
        while k < 50:
            if num_gauss > 1:
                gauss_distribution.append(np.random.randint(1, num_gauss))
            else:
                gauss_distribution.append(1)
            k += 1

        for j in range(0, num_gauss):
            # Randomly create mean and width for the data
            mu = (np.random.randint(-180, 180), np.random.randint(-90, 90))
            width = np.random.uniform(0.0, 30.0)
            cov = np.array([[width, np.random.uniform(-10.0, 10.0)], [np.random.uniform(-10.0, 10.0), width]])
            num_vals_train = gauss_distribution.count(j + 1)
            num_vals_test = int(num_vals_train / 10)
            # Actually create the random samples
            samples_train = np.random.multivariate_normal(mu, cov, num_vals_train).round(5)
            samples_test = np.random.multivariate_normal(mu, cov, num_vals_test)
            samples_train, samples_test = wrap_around(samples_train, samples_test)
            # For the first values, ensure we don't have an empty array
            if x_train.size == 2:
                x_train = samples_train
                y_train = [i] * num_vals_train
                x_val = samples_test
                y_val = [i] * num_vals_test
            else:
                # Add new values to previous
                x_train = np.concatenate([x_train, samples_train])
                y_train = np.concatenate([y_train, [i] * num_vals_train])
                x_val = np.concatenate([x_val, samples_test])
                y_val = np.concatenate([y_val, [i] * num_vals_test])

    return x_train, y_train, x_val, y_val

# Create 'sightings' of species, mimicking iNat
def create_presences(NUM_CLASS, NUM_DATA):
    centres = [(-175, -70), (-65, -40), (0, 80), (120, 25), (45, 40)]
    x_train = np.empty((1, 2))
    y_train = np.empty(1)
    x_val = np.empty((1, 2))
    y_val = np.empty(1)

    for i in range(0, NUM_CLASS):
        print("Generating Gaussian for species", i)
        # fig, ax = plt.subplots()
        cluster = np.random.randint(5)
        num_gauss = np.random.randint(10)
        migratory = np.random.randint(100)

        if (num_gauss < 4):
            num_gauss = 1
        elif (num_gauss < 7):
            num_gauss = 2
        elif (num_gauss < 9):
            num_gauss = 3
        else:
            num_gauss = 4

        for j in range(0, num_gauss):
            # Random values for generating random spreads of data points
            if (migratory == 0):
                mu = (np.random.randint(-180, 180), np.random.randint(-90, 90))
            else:
                xAdd = np.random.uniform(-30.0, 30.0)
                yAdd = np.random.uniform(-30.0, 30.0)
                mu = (centres[cluster][0] + xAdd, centres[cluster][1] + yAdd)
            width = np.random.uniform(0.0, 30.0)
            cov = np.array([[width, np.random.uniform(-10.0, 10.0)], [np.random.uniform(-10.0, 10.0), width]])
            num_vals_train = np.random.randint(NUM_DATA)
            num_vals_test = int(num_vals_train / 10) + 1
            # Generate the random data points
            samples_train = np.random.multivariate_normal(mu, cov, num_vals_train).round(5)
            samples_test = np.random.multivariate_normal(mu, cov, num_vals_test)
            samples_train, samples_test = wrap_around(samples_train, samples_test)
            # Ensure we don't have an empty array
            if x_train.size == 2:
                x_train = samples_train
                y_train = [i] * num_vals_train
                x_val = samples_test
                y_val = [i] * num_vals_test
            else:
                # Add new data points to previous
                x_train = np.concatenate([x_train, samples_train])
                y_train = np.concatenate([y_train, [i] * num_vals_train])
                x_val = np.concatenate([x_val, samples_test])
                y_val = np.concatenate([y_val, [i] * num_vals_test])

    return x_train, y_train, x_val, y_val

# Generate pseudo-negatives, random data points around the grid
def create_absences(NUM_CLASS, NUM_ABS):
    x_absences = np.empty((NUM_CLASS * NUM_ABS, 2)).astype(np.float32)
    y_absences = np.empty((NUM_CLASS * NUM_ABS, 1))
    for i in range(0, NUM_CLASS):
        for j in range(0, NUM_ABS):
            (x, y) = (np.random.randint(-180, 180), np.random.randint(-90, 90))
            x_absences[(i * NUM_ABS) + j] = [x, y]
            y_absences[(i * NUM_ABS) + j] = i
    return x_absences, y_absences

# Create a graph of a given class, as seen in the report
def graph_class(train, y, range, map):
    im_width  = (range[1] - range[0]) // 45  # 8
    im_height = (range[3] - range[2]) // 45  # 4
    plt.figure(num=0, figsize=[im_width, im_height])
    filtered = train.loc[(train['y_train'] == y)]
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((-180, 180))
    plt.ylim((-90, 90))
    plt.title("Distribution of Class " + str(y))
    plt.scatter(filtered['x_train0'], filtered['x_train1'])
    if map:
        plt.imshow(np.load('ocean_mask.npy').astype(np.int), cmap='gray', extent=range)
    plt.show()

# Code for initialising the synthetic data
def init_synth_dataset(NUM_CLASS, NUM_DATA, MAP_RANGE, NUM_ABS, mini=True):
    # Create data
    if mini:
        x_train, y_train, x_val, y_val = create_presences_equal(NUM_CLASS)
    else:
        x_train, y_train, x_val, y_val = create_presences(NUM_CLASS, NUM_DATA)
    x_absences, y_absences = create_absences(NUM_CLASS, NUM_ABS)

    # Add to dataframes
    dfTrain = pd.DataFrame(list(zip(x_train[:, 0], x_train[:, 1], y_train)),
                           columns=['x_train0', 'x_train1', 'y_train'])
    dfTest = pd.DataFrame(list(zip(x_val[:, 0], x_val[:, 1], y_val)), columns=['x_val0', 'x_val1', 'y_val'])
    dfAbsences = pd.DataFrame(list(zip(x_absences[:, 0], x_absences[:, 1], y_absences)),
                              columns=['x_absences0', 'x_absences1', 'y_absences'])

    # Visualise class 0 to confirm correctness
    graph_class(dfTrain, 0, MAP_RANGE, False)

    # Write data to files
    if mini:
        write_to_files(dfTrain, dfTest, dfAbsences, 'synth_data/synth_data_' + str(NUM_CLASS) + '_mini')
    else:
        write_to_files(dfTrain, dfTest, dfAbsences, 'synth_data/synth_data_' + str(NUM_CLASS) + '_full')

    return dfTrain, dfTest, dfAbsences


def init_inat_dataset(NUM_CLASS, MAP_RANGE, NUM_ABS, mini=True):
    # Read in raw data from either mini or full dataset
    if mini:
        dataTrain = json.load(open('inat2021_data/train_mini.json'))
    else:
        dataTrain = json.load(open('inat2021_data/train.json'))
    dataVal = json.load(open('inat2021_data/val.json'))
    # Extract the values we are interested in to an array
    x_train = [[ii['id'], ii['longitude'], ii['latitude']] for ii in dataTrain['images']]
    x_val = [[ii['id'], ii['longitude'], ii['latitude']] for ii in dataVal['images']]
    x_absences, y_absences = create_absences(NUM_CLASS, NUM_ABS)

    # Catch missing values so we can remove them later
    corrupt_train = []
    for i in x_train:
        if (i[1] is None or i[2] is None):
            corrupt_train.append(i[0])

    corrupt_val = []
    for i in x_val:
        if i[1] is None or i[2] is None:
            corrupt_val.append(i[0])

    # Remove corrupt (missing) values from data
    x_train = [[ii[0], ii[1], ii[2]] for ii in x_train if ii[0] not in corrupt_train]
    x_val = [[ii[0], ii[1], ii[2]] for ii in x_val if ii[0] not in corrupt_val]
    y_train = [[ii['id'], ii['category_id']] for ii in dataTrain['annotations'] if ii['id'] not in corrupt_train]
    y_val = [[ii['id'], ii['category_id']] for ii in dataVal['annotations'] if ii['id'] not in corrupt_val]

    print(len(x_train), len(y_train))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    # Create dataframes from the filtered dataset
    dfTrain = pd.DataFrame(list(zip(x_train[:, 1], x_train[:, 2], y_train[:, 1])),
                           columns=['x_train0', 'x_train1', 'y_train'])
    dfVal = pd.DataFrame(list(zip(x_val[:, 1], x_val[:, 2], y_val[:, 1])), columns=['x_val0', 'x_val1', 'y_val'])
    dfAbsences = pd.DataFrame(list(zip(x_absences[:, 0], x_absences[:, 1], y_absences)),
                              columns=['x_absences0', 'x_absences1', 'y_absences'])

    # Trim the dataset to only the number of classes we are interested in
    dfTrain = dfTrain[dfTrain['y_train'] < NUM_CLASS]
    dfVal = dfVal[dfVal['y_val'] < NUM_CLASS]

    # Write to file
    if mini:
        write_to_files(dfTrain, dfVal, dfAbsences, 'inat_data/inat_data_' + str(NUM_CLASS) + '_mini')
    else:
        write_to_files(dfTrain, dfVal, dfAbsences, 'inat_data/inat_data_' + str(NUM_CLASS) + '_full')

    return dfTrain, dfVal, dfAbsences
