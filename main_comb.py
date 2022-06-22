#!/usr/bin/env python3
from re import L
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import csv
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from models.resnet.extract_resnet import ExtractResNet
from omegaconf import OmegaConf
from utils.utils import build_cfg_path, form_list_from_user_input, sanity_check
#set random seeds
class SUBNET(nn.Module):
    def __init__(self,num_classes):
        super(SUBNET, self).__init__()
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        # x represents our data
    def forward(self, x):
        # Pass data through conv1
        x = self.dropout1(x)
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through fc1
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x_s = self.fc(x)
        # Apply softmax to x
        #   output = x.reshape(300,128)

        return x, torch.sigmoid(x_s)
class VideoDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,data,label,extractor):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.label = label
        self.extractor = extractor
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with torch.no_grad():
            im = self.extractor(self.data[idx])
        lb = self.label[idx]
        im = torch.Tensor(im)
        lb = torch.Tensor(np.array(lb,dtype=np.float16))   
        return (im, lb)
if __name__ == "__main__":
    cfg_cli = OmegaConf.from_cli()
    print(cfg_cli)
    cfg_yml = OmegaConf.load(build_cfg_path(cfg_cli.feature_type))
    # the latter arguments are prioritized
    cfg = OmegaConf.merge(cfg_yml, cfg_cli)
    # OmegaConf.set_readonly(cfg, True)
    print(OmegaConf.to_yaml(cfg))
    # some printing
    if cfg.on_extraction in ['save_numpy', 'save_pickle']:
        print(f'Saving features to {cfg.output_path}')
    if cfg.keep_tmp_files:
        print(f'Keeping temp files in {cfg.tmp_path}')

   
    seed=42
    torch.manual_seed(seed)
    writer = SummaryWriter()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters

    num_classes = 1
    num_epochs = 2
    batch_size = 64
    learning_rate = 0.0001
    learning_rate_decay = 0.95

    input_size = 128
    sequence_length = 30
    hidden_size = 2000
    num_layers = 3
    loss_hyp=0.8



    org_path = os.getcwd()
    #read csv
    rows = []
    rows_test = []
    with open("bc_detection_train.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)
    with open("bc_detection_val.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows_test.append(row)
    print(header)
    print(rows_test[:][0])
    #reading input data
    path = '/Volumes/maqsood/SSP/main/data/'
    path1= '/Volumes/maqsood/SSP/main/data/'


    combined_data_video = np.array([path+fname[0]+'_video.avi' for fname in rows])
    # combined_data_audio = np.array([path1+fname[0]+'_audio_vggish.wav' for fname in rows])
    labels = np.array([np.array(fname[1], dtype=np.float16) for fname in rows])
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    combined_data_test_video = np.array([path+fname[0]+'_video.avi' for fname in rows_test])
    # combined_data_test_audio = np.array([path1+fname[0]+'_audio_vggish.wav' for fname in rows_test])
    labels_test = np.array([np.array(fname[1], dtype=np.float16) for fname in rows_test])
    # os.chdir(org_path)
    tensor_x_video = combined_data_video # transform to torch tensor
    # tensor_x_audio = combined_data_audio # transform to torch tensor





    tensor_x_test_video = combined_data_test_video # transform to torch tensor
    # tensor_x_test_audio =combined_data_test_audio # transform to torch tensor
    tensor_y = torch.Tensor(labels.astype(np.float64))
    tensor_y_test = torch.Tensor(labels_test.astype(np.float64))
    extractor = ExtractResNet(cfg)
    for param in extractor.parameters():
        param.requires_grad = False

    my_dataset_video_test = VideoDataset(tensor_x_test_video,tensor_y_test,extractor)
    # my_dataset_audio_test = VideoDataset(tensor_x_test_audio,tensor_y_test,extractor = ExtractResNet(cfg))


    my_dataset_video = VideoDataset(tensor_x_video,tensor_y,extractor)
    # my_dataset_audio = VideoDataset(tensor_x_audio,tensor_y,extractor = ExtractResNet(cfg))
    #dataset_train_video, dataset_validate_video = train_test_split(
    #        my_dataset_video, test_size=0.20, random_state=84 #0.02
    #    )
    #dataset_train_audio, dataset_validate_audio = train_test_split(
    #        my_dataset_audio, test_size=0.20, random_state=84 #0.02
    #    )
    #print(f'shape of train:{len(dataset_train_video)}')
    #print(f'shape of validate:{len(dataset_validate_video)}')
    #print(f'shape of test:{len(my_dataset_video_test)}')
    #---------------------
    # class ConcatDataset(torch.utils.data.Dataset):
    #     def __init__(self, *datasets):
    #         self.datasets = datasets

    #     def __getitem__(self, i):
    #         return tuple(d[i] for d in self.datasets)

    #     def __len__(self):
    #         return min(len(d) for d in self.datasets)

    my_dataloader = torch.utils.data.DataLoader(
                    my_dataset_video,batch_size=batch_size,num_workers = 4, prefetch_factor= 10, persistent_workers = True, shuffle=True)
    #my_dataloader_val = torch.utils.data.DataLoader(
    #             ConcatDataset(
    #                 dataset_validate_video,
    #                 dataset_validate_audio
    #             ),
    #             batch_size=batch_size)
    my_dataloader_test = torch.utils.data.DataLoader(
                    my_dataset_video_test,
                batch_size=batch_size,num_workers = 4, prefetch_factor= 10, persistent_workers = True)
    #---------------------




    




    #Fully connected neural network with one hidden layer
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(RNN, self).__init__()
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            #self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            # -> x needs to be: (batch_size, seq, input_size)

            # or:
            #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            # self.fc_enc = SUBNET()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
            
        def forward(self, x):
            # Set initial hidden states (and cell states for LSTM)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

            # x: (n, 28, 28), h0: (2, n, 128)

            # Forward propagate RNN
            #out, _ = self.rnn(x, h0)
            # or:
            # x = self.fc_enc(x)
            out, _ = self.lstm(x, (h0,c0))

            # out: tensor of shape (batch_size, seq_length, hidden_size)
            # out: (n, 28, 128)

            # Decode the hidden state of the last time step
            out = out[:, -1, :]
            # out: (n, 128)

            out = self.fc(out)
            # out: (n, 10)
            return torch.sigmoid(out)
    
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    #model_trans = fusion.TransformerEncoder(num_layers = 4,input_dim =128,num_heads =4, dim_feedforward = 256)
    model_fc = SUBNET(num_classes).to(device)
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_fc = nn.BCELoss()
    #criterion_trans = nn.BCELoss()
    # optimizer_fc = torch.optim.Adam(model_fc.parameters(), lr=learning_rate)
    # Train the model
    best_val_acc = 0
    for epoch in range(num_epochs):
        correct = 0
        num_samples = 0
        model.train()
        model_fc.train()
        for i, (dataset1, dataset2) in enumerate(my_dataloader):
            # origin shape: [N, 1, 28, 28]
            # resized: [N, 300, 2048][N,300,128]
            data1 = dataset1.to(device)
            labels = dataset2.to(device)
            # data2 = dataset2[0].to(device)
            # label2 = dataset2[1].to(device)
            # Forward pass
            images = torch.as_tensor([]).to(device)
            loss_fc = 0
            #loss_trans = 0
            for k in range(data1.size(0)):
                images_o,outputs_fc_s = model_fc(data1[k])
                images = torch.cat((images,images_o))
                loss_fc += criterion_fc(outputs_fc_s, labels[k].reshape(1,1).expand(sequence_length, 1))
                # Backward and optimize
                # optimizer_fc.zero_grad()
                # loss_fc.backward()
                # optimizer_fc.step()
            # print(f'outputs.shape:{outputs.shape}')
            # print(labels.shape)
            # outputs = outputs.squeeze()
            

            loss_fc = loss_fc/data1.size(0)
            images = images.reshape(-1, sequence_length, input_size).to(device)
            # print(images.shape)
            labels = labels.to(device)
            num_samples+=labels.size(0)
            # Forward pass
            outputs = model(images)
            # print(f'outputs.shape:{outputs.shape}')
            # print(labels.shape)
            # outputs = outputs.squeeze()
            loss = criterion(outputs, labels.unsqueeze(1))
            loss +=loss_hyp*loss_fc
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predicted = (outputs > 0.5).long()
            # print(f'predicted.shape:{predicted.shape}')
            # print(f'labels.shape:{labels.shape}')
            # print(f'correct_pre:{correct}')
            correct += (predicted.squeeze()== labels).sum().item()
        
        print('[%d/%d] loss: %.3f, accuracy: %.3f' %
            (i , epoch, loss.item(), 100 * correct /num_samples))
        writer.add_scalars('Loss',{'train':loss.item()},epoch)
        writer.add_scalars('Accuracy', {'train': 100 * correct /num_samples},epoch)
        learning_rate *= learning_rate_decay
        update_lr(optimizer, learning_rate)
        # Test the model
        # In test phase, we don't need to compute gradients (for memory efficiency)
        num_samples_val = 0
        model.eval()
        model_fc.eval()
        #model_trans.eval()
        with torch.no_grad():
            correct_val = 0
            for i, (dataset1, dataset2) in enumerate(my_dataloader_test):
            # origin shape: [N, 1, 28, 28]
            # resized: [N, 300, 2048][N,300,128]
                data1 = dataset1.to(device)
                labels = dataset2.to(device)
                # Forward pass
                images = torch.as_tensor([]).to(device)
                loss_fc = 0
                for k in range(data1.size(0)):
                    images_o,outputs_fc_s = model_fc(data1[k])
                
                    loss_fc += criterion_fc(outputs_fc_s, labels[k].reshape(1,1).expand(sequence_length, 1))
                loss_fc = loss_fc/data1.size(0)
                images = images.reshape(-1, sequence_length, input_size).to(device)
                labels = labels.to(device)
                num_samples_val+=labels.size(0)
                outputs = model(images)
                predicted = (outputs > 0.5).long()
                correct_val += (predicted.squeeze()== labels).sum().item()

            val_acc = 100 * correct_val / num_samples_val
            print(f'Accuracy of the network on the validation: {val_acc} %')
            writer.add_scalars('Accuracy', {'val': val_acc},epoch)
        if(val_acc> best_val_acc):
            best_val_acc = val_acc
            torch.save(model.state_dict(),'./best_model'+'.ckpt')
            torch.save(model.state_dict(),'./best_model_fc'+'.ckpt')
            print("best model with val acc "+ str(best_val_acc)+ "is saved")
    model.eval()
    model_fc.eval()
    model.load_state_dict(torch.load('./best_model.ckpt'))   
    with torch.no_grad():
            correct_val = 0
            num_samples_val = 0
            for i, (dataset1, dataset2) in enumerate(my_dataloader_test):
            # origin shape: [N, 1, 28, 28]
            # resized: [N, 300, 2048][N,300,128]
                data1 = dataset1.to(device)
                labels = dataset2.to(device)
                # Forward pass
                images = torch.as_tensor([]).to(device)
                loss_fc = 0
                for k in range(data1.size(0)):
                    images_o,_ = model_fc(data1[k])
                    images = torch.cat((images,images_o))
                    # loss_fc += criterion_fc(outputs_fc_s, labels.unsqueeze(1).expand(sequence_length, 1))

                images = images.reshape(-1, sequence_length, input_size).to(device)
                labels = labels.to(device)
                num_samples_val+=labels.size(0)
                outputs = model(images)
                predicted = (outputs > 0.5).long()
                correct_val += (predicted.squeeze()== labels).sum().item()

            val_acc = 100 * correct_val / num_samples_val
            print(f'Accuracy of the network on the test: {val_acc} %')
