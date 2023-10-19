from utils import *
import torch

path = '/expanse/projects/nsg/external_users/public/arno/'
winLength = 2
numChan = 128
srate = 128
feature = 'raw'
one_channel = False

role = 'train'
train_data = load_data_mixed(path + "child_mind_abdu128/", role, winLength, numChan, srate, feature, one_channel)
print(f'X_train shape: {len(train_data)}, {train_data[0][0].shape}')
print(f'Y_train shape: {len(train_data)}, {train_data[0][1].shape}')

role = 'val'
val_data = load_data_mixed(path + "child_mind_abdu128/", role, winLength, numChan, srate, feature, one_channel)
print(f'X_val shape: {len(val_data)}, {val_data[0][0].shape}')
print(f'Y_val shape: {len(val_data)}, {val_data[0][1].shape}')

role = 'test'
test_data = load_data_constant(path + "child_mind_abdu128/", role, winLength, 24, srate, feature, one_channel)
print(f'X_test shape: {len(test_data)}, {test_data[0][0].shape}')
print(f'Y_test shape: {len(test_data)}, {test_data[0][1].shape}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def collate_batch(batch):
    y_list = []
    x_list = []
    for x, y in batch:
        y_list.append(y)
        x_list.append(torch.from_numpy(x[0]))
    y_tensor = torch.tensor(y_list)
    return x_list, y_tensor

batch_size = 70 
loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn = collate_batch)
loader_val = DataLoader(val_data, batch_size=batch_size, collate_fn = collate_batch)
loader_test = DataLoader(test_data, batch_size=batch_size, collate_fn = collate_batch)

for s in range(10):
    model = run_experiment(s, loader_train, loader_val, loader_test, 15)
    print('Next seed loading...')



