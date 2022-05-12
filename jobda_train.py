'''
Implementation of paper 'Joint-Label Learning by Dual Augmentation for Time Series Classification'
Author: Zhengke Sun
'''

import sys
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from model.resnext import ResNeXt
from model.jobda import tws,tws_item

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def main():

    ## load dataset

    x_dir = './data/x_train.npy'
    y_dir = './data/y_train.npy'

    x_data = np.load(x_dir)
    x_data = torch.tensor(x_data)
    x_data = x_data.transpose(1, 2)

    y_data = np.load(y_dir)
    y_data = torch.tensor(y_data, dtype=torch.long)
    y_data = torch.argmax(y_data, dim=1)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=1/5, random_state=7)

    # X = torch.tensor(x_train)
    # y = torch.tensor(y_train, dtype=torch.long)
    #
    # X_val = torch.tensor(x_test)
    # y_val = torch.tensor(y_test, dtype=torch.long)

    X = x_train.clone().detach()
    y = y_train.clone().detach()

    X_val = x_test.clone().detach()
    y_val = y_test.clone().detach()

    ## build model,criterion,optimizer,dataloader
    # Initail paramaters

    N_list=[4]
    n = len(N_list)+1
    num_classes = 7

    # Initial Model,Transform
    model = ResNeXt(8, 29, num_classes*n, 64, 4)
    transform = tws(tws_item, N_list=N_list)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())

    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128,
                              shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(dataset=val_dataset, batch_size=128,
                            shuffle=True)

    val_num = len(val_dataset)
    train_steps = len(train_loader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    save_path = './output/ResNext_jobda_1d.pth'

    epochs = 20
    best_acc = 0.0

    ## train and validation
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for batch_x, batch_y in train_bar:

            batch_size = batch_x.shape[0]
            batch_x = transform(batch_x)
            n = batch_x.shape[0] // batch_size

            logits = model(batch_x.to(device))
            labels = torch.stack([batch_y * n + i for i in range(n)], 0).view(-1)
            loss = criterion(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] total_loss:{:.5f}".format(epoch + 1,
                                                                           epochs,
                                                                           loss)

        model.eval()
        acc = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)

            for val_batch_x, val_batch_y in val_bar:

                val_batch_size = val_batch_x.shape[0]
                ds = nn.Conv1d(in_channels= val_batch_size, out_channels= val_batch_size,
                               kernel_size=n, stride=n,
                               groups= val_batch_size,
                               padding=0, bias=False)
                ds.weight.data = torch.tensor([[[1.] * n]] * val_batch_size)
                ds.to(device)

                outputs = model(val_batch_x.to(device))

                outputs = outputs.view([1,val_batch_size,num_classes*n])
                outputs = ds(outputs)
                outputs = outputs.view([val_batch_size,num_classes])

                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_batch_y.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.5f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

    print('Finished Training')

if __name__ == '__main__':
    main()