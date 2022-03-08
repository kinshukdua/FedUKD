import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.transforms as transforms
from losses import FocalLoss, mIoULoss
from model import UNet, Custom_Slim_UNet
from dataloader import segDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# T = 5 is best
T = 5

# alpha = 0.8 is best, 0.9 is too much, less than 0.8 decreases accuracy
alpha = 0.8


def pixel_wise_loss(student_output, teacher_output):
    _,C,W,H = student_output.shape

    # what would happen if we use softmax?
    # Softmax didn't work
    pred_T = torch.sigmoid(teacher_output/T)
    pred_S = torch.sigmoid(student_output/T).log()

    #criterion = torch.nn.KLDivLoss(reduction = 'batchmean')
    #KLDloss = - criterion(pred_S, pred_T)
    #TODO: map this to KLDL
    #KDloss = - sum(p * log (p/q)) ---> refer notes page 15 - 16 
    #Pixelwise loss = sum(-p*logq)
    #KLDiv = relative entropy
    pixelwise_loss = (- pred_T * pred_S)
    # Check to see if it is needed
    return  torch.sum(pixelwise_loss) / (W*H*10)

def loss_fn(student_output, teacher_output, gt , criterion):
    '''student_output = student_output.round() 
    student_output[student_output<0] = 0
    gt = torch.clamp(gt, min = 0, max = 1)
    teacher_output = torch.clamp(teacher_output, min = 0, max = 1)'''

    student_output = student_output.clamp(min = 0, max = 1)
    teacher_output = teacher_output.clamp(min = 0, max = 1)
    student_loss = criterion(student_output, gt)
    kd_loss = pixel_wise_loss(student_output, teacher_output)
    #not sure about using T, also check KLD
    loss = (student_loss*(1-alpha) + (kd_loss)*(alpha)) # as per structured KD paper
    return loss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='path to your dataset')
    parser.add_argument('--test', type=str, help='path to your test dataset')
    parser.add_argument('--meta', type=str, required=True, help='path to your metadata')
    parser.add_argument('--name', type=str, default="unet", help='name to be appended to checkpoints')
    parser.add_argument('--num_epochs', type=int, default=100, help='dnumber of epochs')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--loss', type=str, default='focalloss', help='focalloss | iouloss | crossentropy')
    return parser.parse_args()

def acc(label, predicted):
    seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(y.cpu())
    return seg_acc

if __name__ == '__main__':
    args = get_args()
    N_EPOCHS = args.num_epochs
    BACH_SIZE = args.batch

    color_shift = transforms.ColorJitter(.1,.1,.1,.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

    t = transforms.Compose([color_shift, blurriness])
    dataset = segDataset(args.data, args.meta, training = True, transform= t)
    n_classes = len(dataset.bin_classes)+1

    print('Number of data : '+ str(len(dataset)))

    if not args.test : 
        dataset = segDataset(args.data, args.meta, training = True, transform= t)
        n_classes = len(dataset.bin_classes)+1
        print('Number of data : '+ str(len(dataset)))
        test_num = int(0.1 * len(dataset))
        print(f'Test data : {test_num}')
        print(f"Number of classes : {n_classes}")
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-test_num, test_num], generator=torch.Generator().manual_seed(101))
        N_DATA, N_TEST = len(train_dataset), len(test_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BACH_SIZE, shuffle=True, num_workers=2)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=1)
    else :
        dataset = segDataset(args.data, args.meta, training = True, transform= t)
        dataset2 = segDataset(args.test, args.meta, training = False, transform= t)
        n_classes = len(dataset.bin_classes)+1
        print('Number of train data : '+ str(len(dataset)))
        test_num = len(dataset2)
        print(f'Test data : {test_num}')
        print(f"Number of classes : {n_classes}")
        N_DATA, N_TEST = len(dataset), len(dataset2)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BACH_SIZE, shuffle=True, num_workers=2)
        test_dataloader = torch.utils.data.DataLoader(dataset2, batch_size=BACH_SIZE, shuffle=False, num_workers=1)

    if args.loss == 'focalloss':
        criterion = FocalLoss(gamma=3/4).to(device)
    elif args.loss == 'iouloss':
        criterion = mIoULoss(n_classes=n_classes).to(device)
    elif args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        print('Loss function not found!')

    model = UNet(n_channels=3, n_classes=n_classes, bilinear=True, channel_depth=8).to(device)
    teacher = UNet(n_channels=3, n_classes=n_classes, bilinear=True).to(device)
    teacher.load_state_dict(torch.load('C:\\projects\\LandUse\\Semantic-Segmentation-UNet-Federated-master\\saved_models\\BigCS_epoch_1_1.26202.pt'))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    min_loss = torch.tensor(float('inf'))
    os.makedirs('./saved_models_small', exist_ok=True)

    plot_losses = []
    scheduler_counter = 0

    for epoch in range(N_EPOCHS):
        # training
        model.train()
        # teacher.eval()
        loss_list = []
        acc_list = []
        for batch_i, (x, y) in enumerate(train_dataloader):
            with torch.no_grad():
                teach_mask = teacher(x.to(device))
            pred_mask = model(x.to(device))  
            loss = loss_fn(pred_mask, teach_mask, y.to(device), criterion)
            # loss = criterion(pred_mask.to(device),y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.cpu().detach().numpy())
            acc_list.append(acc(y,pred_mask).numpy())

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f)]"
                % (
                    epoch,
                    N_EPOCHS,
                    batch_i,
                    len(train_dataloader),
                    loss.cpu().detach().numpy(),
                    np.mean(loss_list),
                )
            )
        scheduler_counter += 1
        # testing
        model.eval()
        val_loss_list = []
        val_acc_list = []
        for batch_i, (x, y) in enumerate(test_dataloader):
            with torch.no_grad():    
                pred_mask = model(x.to(device))  
            val_loss = criterion(pred_mask, y.to(device))
            val_loss_list.append(val_loss.cpu().detach().numpy())
            val_acc_list.append(acc(y,pred_mask).numpy())
            
        print(' epoch {} - loss : {:.5f} - acc : {:.2f} - val loss : {:.5f} - val acc : {:.2f}'.format(epoch, 
                                                                                                        np.mean(loss_list), 
                                                                                                        np.mean(acc_list), 
                                                                                                        np.mean(val_loss_list),
                                                                                                        np.mean(val_acc_list)))
        plot_losses.append([epoch, np.mean(loss_list), np.mean(val_loss_list)])

        compare_loss = np.mean(val_loss_list)
        is_best = compare_loss < min_loss
        if is_best == True:
            scheduler_counter = 0
            min_loss = min(compare_loss, min_loss)
            torch.save(model.state_dict(), './saved_models/{}_epoch_{}_{:.5f}.pt'.format(args.name,epoch,np.mean(val_loss_list)))
        
        if scheduler_counter > 5:
            lr_scheduler.step()
            print(f"lowering learning rate to {optimizer.param_groups[0]['lr']}")
            scheduler_counter = 0


    # plot loss
    plot_losses = np.array(plot_losses)
    plt.figure(figsize=(12,8))
    plt.plot(plot_losses[:,0], plot_losses[:,1], color='b', linewidth=4)
    plt.plot(plot_losses[:,0], plot_losses[:,2], color='r', linewidth=4)
    plt.title(args.loss, fontsize=20)
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('loss',fontsize=20)
    plt.grid()
    plt.legend(['training', 'validation']) # using a named size
    plt.savefig('loss_plots.png')

