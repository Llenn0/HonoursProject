from enum import Enum
import numpy as np
import sklearn.metrics
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt
import net
import dataset
# import transfer

# Define the different modes we can run in
class Mode(Enum):
    INIT_SYNTH = 1
    INIT_SYNTH_MINI = 2
    INIT_INAT = 3
    INIT_INAT_MINI = 4
    LOAD_SYNTH = 5
    LOAD_SYNTH_MINI = 6
    LOAD_INAT = 7
    LOAD_INAT_MINI = 8
    # TRANSFER = 9
    # TRANSFER_MINI = 10

# These are the main parameters to change with each experiment
NUM_CLASS = 100
MAPPING_TYPE = 'fourier'
MODE = Mode.LOAD_SYNTH

# Hyperparameters for dataset generation - not changed often
NUM_DATA = 240
NUM_ABS = 10
# RESULTS_DIR = None
# COMPARISON = None

B = None
showMap = False # If enabled, shows a map when graphing a class. Usually enabled for iNat and disabled for synthetic.
DIM_HIDDEN = 256
MAP_RANGE = (-180, 180, -90, 90)
cls = 10 # The class to show as a graph

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Create B matrix from hyperparameters
if MAPPING_TYPE == 'fourier':
    SIZE = 128
    SCALE = 2
    B = np.random.randn(SIZE, 2) * SCALE
    # RESULTS_DIR = 'results_fourier_encoding'
    # COMPARISON = 'lat_lon_gauss_basis'
elif MAPPING_TYPE == 'sincos':
    SIZE = 2
    SCALE = 0
    B = np.eye(2)
    # RESULTS_DIR = 'results_basic_encoding'
    # COMPARISON = 'lat_lon_wrapped'
else:
    SIZE = 1
    SCALE = 0
    # RESULTS_DIR = 'results_base'
    # COMPARISON = 'lat_lon'

# Remove any extra tokens from loaded datasets
def clean_list(list):
    new_list = np.zeros(list.shape, dtype=np.float32)
    for i in range(0, len(list)):
        new_list[i] = int(list[i].strip("[]."))
    return new_list

if MODE == Mode.LOAD_SYNTH:
    dfTrain = pd.read_csv('synth_data/synth_data_'+str(NUM_CLASS)+'_full/train.csv')
    dfVal = pd.read_csv('synth_data/synth_data_'+str(NUM_CLASS)+'_full/val.csv')
    dfAbsences = pd.read_csv('synth_data/synth_data_'+str(NUM_CLASS)+'_full/absences.csv')
elif MODE == Mode.LOAD_SYNTH_MINI:
    dfTrain = pd.read_csv('synth_data/synth_data_'+str(NUM_CLASS)+'_mini/train.csv')
    dfVal = pd.read_csv('synth_data/synth_data_'+str(NUM_CLASS)+'_mini/val.csv')
    dfAbsences = pd.read_csv('synth_data/synth_data_'+str(NUM_CLASS)+'_mini/absences.csv')
elif MODE == Mode.INIT_INAT:
    dfTrain, dfVal, dfAbsences = dataset.init_inat_dataset(NUM_CLASS, MAP_RANGE, NUM_ABS, mini=False)
    exit()
elif MODE == Mode.INIT_INAT_MINI:
    dfTrain, dfVal, dfAbsences = dataset.init_inat_dataset(NUM_CLASS, MAP_RANGE, NUM_ABS)
    exit()
elif MODE == Mode.LOAD_INAT:
    dfTrain = pd.read_csv('inat_data/inat_data_'+str(NUM_CLASS)+'_full/train.csv')
    dfVal = pd.read_csv('inat_data/inat_data_'+str(NUM_CLASS)+'_full/val.csv')
    dfAbsences = pd.read_csv('inat_data/inat_data_'+str(NUM_CLASS)+'_full/absences.csv')
    showMap = True
elif MODE == Mode.LOAD_INAT_MINI:
    dfTrain = pd.read_csv('inat_data/inat_data_'+str(NUM_CLASS)+'_mini/train.csv')
    dfVal = pd.read_csv('inat_data/inat_data_'+str(NUM_CLASS)+'_mini/val.csv')
    dfAbsences = pd.read_csv('inat_data/inat_data_'+str(NUM_CLASS)+'_mini/absences.csv')
    showMap = True
elif MODE == Mode.INIT_SYNTH:
    dfTrain, dfVal, dfAbsences = dataset.init_synth_dataset(NUM_CLASS, NUM_DATA, MAP_RANGE, NUM_ABS, mini=False)
    exit()
elif MODE == Mode.INIT_SYNTH_MINI:
    dfTrain, dfVal, dfAbsences = dataset.init_synth_dataset(NUM_CLASS, NUM_DATA, MAP_RANGE, NUM_ABS)
    exit()
# elif MODE == Mode.TRANSFER:
#     dfTrain = pd.read_csv('inat_data/inat_data_'+str(NUM_CLASS)+'_full/train.csv')
#     dfVal = pd.read_csv('inat_data/inat_data_'+str(NUM_CLASS)+'_full/val.csv')
#     dfAbsences = pd.read_csv('inat_data/inat_data_'+str(NUM_CLASS)+'_full/absences.csv')
#     showMap = True
# elif MODE == Mode.TRANSFER_MINI:
#     dfTrain = pd.read_csv('inat_data/inat_data_'+str(NUM_CLASS)+'_mini/train.csv')
#     dfVal = pd.read_csv('inat_data/inat_data_'+str(NUM_CLASS)+'_mini/val.csv')
#     dfAbsences = pd.read_csv('inat_data/inat_data_'+str(NUM_CLASS)+'_mini/absences.csv')
#     showMap = True
else:
    exit()

# Extract arrays from dataframes
x_train = np.array(list(zip(dfTrain['x_train0'], dfTrain['x_train1'])), dtype=np.float32)
y_train = np.array(list(dfTrain['y_train']))
x_val = np.array(list(zip(dfVal['x_val0'], dfVal['x_val1'])), dtype=np.float32)
y_val = np.array(list(dfVal['y_val']))
x_absences = np.array(list(zip(dfAbsences['x_absences0'], dfAbsences['x_absences1'])), dtype=np.float32)
y_absences = np.array(list(dfAbsences['y_absences']))

# plt.scatter(x_absences[0:10, 0], x_absences[0:10, 1])
# plt.title("Absences for Synthetic Class 0")
# plt.show()

# PREPROCESSING
tensor_xtrain = dataset.convert_loc_to_tensor(x_train, B=B, mapping=MAPPING_TYPE, device=device)
tensor_xval = dataset.convert_loc_to_tensor(x_val, B=B, mapping=MAPPING_TYPE, device=device)
tensor_xabsences = dataset.convert_loc_to_tensor(x_absences, B=B, mapping=MAPPING_TYPE, device=device)
tensor_ytrain = torch.from_numpy(y_train).to(device)
tensor_yval = torch.from_numpy(y_val).to(device)
tensor_yabsences = torch.from_numpy(clean_list(y_absences)).to(device)

# Use tensors to create dataloaders for training and validation
train_dataset = TensorDataset(tensor_xtrain, tensor_ytrain)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=500)
val_dataset = TensorDataset(tensor_xval, tensor_yval)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=500)

# Visualise distribution of the chosen class
dataset.graph_class(dfTrain, cls, MAP_RANGE, showMap)

# Create and train network
nnet = net.createNet(NUM_CLASS, 2*SIZE, DIM_HIDDEN).to(device)

net.trainNet(nnet, train_dataloader, tensor_xabsences, NUM_CLASS, NUM_ABS, device)


# Validate network
correct = 0
total = 0
y_score = np.empty((len(y_val), NUM_CLASS))
y_pred = np.empty((len(y_val)))
y_true = np.empty((len(y_val)))
average_precision = np.empty((NUM_CLASS,))
ind = 0

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in val_dataloader:
        data, labels = data
        # calculate outputs by running images through the network
        outputs = nnet(data.float())
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        labels = labels.cpu()
        outputs = outputs.cpu()
        predicted = predicted.cpu()
        for i in range(len(data)):
            y_score[i+ind] = outputs.numpy()[i]
            y_pred[i+ind] = predicted.numpy()[i]
            y_true[i+ind] = labels.numpy()[i]
        ind += len(data)

# Calculate average precision
for i in range(NUM_CLASS):
    y_mask = np.empty((len(y_val)))
    for j in range(len(y_val)):
        if y_true[j] == i:
            y_mask[j] = 1
        else:
            y_mask[j] = 0
    average_precision[i] = sklearn.metrics.average_precision_score(y_mask, y_score[:,i])

np.set_printoptions(suppress=True)
average_precision = np.average(average_precision)

# Calculate recall
recall = np.array((np.linspace(0, NUM_CLASS-1, NUM_CLASS), sklearn.metrics.recall_score(y_true, y_pred, average=None)))
plt.title("Rate of Recall Decline")
plt.ylabel("Recall")
plt.tick_params(axis='x', which='both', bottom='false', top='false')
# Create the recall plots seen in the report
plt.plot(np.linspace(0, NUM_CLASS-1, NUM_CLASS), np.flip(recall[:, recall[1].argsort()]).T[:,0])
plt.show()

# Get the top 10 and bottom 10 recalls in order and print them
top10 = np.flip(recall[:, recall[1].argsort()][:,-11:-1]).T
bot10 = recall[:, recall[1].argsort()][:,0:10].T

print("Top 10 Recall:\n")
for i in top10:
    print(str(i[1]) + ' - ' + str(i[0]))
print("Bottom 10 Recall:\n")
for i in bot10:
    print(str(i[0]) + ' - ' + str(i[1]))

print("Average Precision: ", average_precision)

mask = np.load('ocean_mask.npy').astype(np.int)

# Decision boundary visualisation. Takes the chosen class and visualises the network's confidence for that class
# over the whole grid.

x_span = np.linspace(-180, 180, 181)
y_span = np.linspace(-90, 90, 181)
xx, yy = np.meshgrid(x_span, y_span)

testdata = dataset.convert_loc_to_tensor(np.c_[xx.ravel(), yy.ravel()], B=B, mapping=MAPPING_TYPE, device=device)
im_width = (MAP_RANGE[1] - MAP_RANGE[0]) // 45  # 8
im_height = (MAP_RANGE[3] - MAP_RANGE[2]) // 45  # 4
plt.figure(num=0, figsize=[im_width, im_height])
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((-180, 180))
plt.ylim((-90, 90))
plt.title("Neural Network Prediction Confidence for class " + str(cls))

pred = nnet(testdata.float())[:,cls].cpu().detach().numpy()
# Put the result into a color plot
pred = pred.reshape(xx.shape)
plt.contourf(xx, yy, pred)
plt.colorbar()
# plt.scatter(x_train[i*50:(i*50)+50,0], x_train[i*50:(i*50)+50,1])
plt.show()

# Get ready to execute the transfer learning code if we are performing transfer learning

# if MODE == Mode.TRANSFER:
#     randNet = net.createNet(NUM_CLASS, 2 * SIZE, DIM_HIDDEN).to(device)
#     randNet.random_weights()
#     randNet.output_feats = True
#     nnet.output_feats = True
#     transfer.transfer(['ELEVATION'], [COMPARISON, 'neural_network_random', 'neural_network_features'], randNet, nnet, NUM_CLASS, RESULTS_DIR, encoding=MAPPING_TYPE)