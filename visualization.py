# TODO filter illustration
# import torchvision

# filters = audio_model.module.conv_layers[0].weight.data.detach().cpu()
# print(filters.size())
# def visu_filters(filters: torch.tensor):
#     """@param[in]   filters n, c, h, w"""
#     grid = torchvision.utils.make_grid(filters, nrow=4, padding=2, normalize=True)
#     plt.figure( figsize=(8,8) )
#     plt.imshow(grid.numpy().transpose((1, 2, 0)))

# visu_filters(filters)
# outs = audio_model.module.conv_layers[0].forward(dataset[4][0].unsqueeze(0).cuda())
# plt.imshow(outs[3,:,:].detach().cpu())

#################################
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

def plot_spectrogram(mel_dataset, index):
    """@brief   plots the mel spectrogram based on the sample form the dataset
    @param[in]  mel_datset  dataset containing the mel spectrograms
    @param[in]  index       index of the sample in the dataset"""
    spect, _ = mel_dataset[index]
    if mel_dataset.use_kaldi:
        spect = spect.transpose(0, 1)
    plt.figure()
    librosa.display.specshow(
        spect.detach().numpy(), cmap='magma', sr=mel_dataset.sample_rate,
        hop_length=mel_dataset.hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f')
    plt.xlabel('Idő [s]')
    plt.ylabel('Frekvencia mel skálán [Hz]')
    title_part = 'augmentációval' if mel_dataset.augment else 'augmentáció nélkül'
    plt.title(f'Spektrogram {title_part}')

def plot_waveform(wave_dataset, index):
    """@brief   plots the waveform based on the sample from the dataset
    @param[in]  wave_dataset    dataset containing the waveforms
    @param[in]  index       index of the sample in the dataset"""
    wave, _ = wave_dataset[index]
    plt.figure()
    plt.plot([i/wave_dataset.sample_rate for i in range(len(wave))],
            wave, color='darkviolet')
    plt.xlabel('Idő [s]')
    plt.ylabel('Amplitúdó', color='darkviolet')
    plt.title('Hullámforma')

def plot_train_proc(last_train_stats, title):
    """@brief   plots validation and running data of the last training
    @param[in]  title   title of the plotted figure"""
    epochs = last_train_stats['epoch']

    _, ax_loss = plt.subplots()
    ax_acc = ax_loss.twinx()
    ax_loss.plot(epochs, last_train_stats['val_loss'], color='darkorange')
    ax_loss.plot(epochs, last_train_stats['avg_train_loss'], color='tomato')
    ax_acc.plot(epochs, last_train_stats['val_acc'], color='darkviolet')
    
    ax_loss.set_title(title)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Hiba', color='tomato')#loss
    ax_acc.set_ylabel('Pontosság', color='darkviolet')#accuracy
    plt.show()

def plot_confusion_matrix(classes_list, start_index, test_loader, audio_model, have_cuda):
    """@brief   computes and plots confusion matrix
    @param[in]  classes_list    list of the classes
    @param[in]  start_index     first index to be illustrated on the matrix 
    @param[in]  test_loader     DataLoader for the test samples
    @param[in]  audio_model     the network to be tested
    @param[in]  have_cuda       if cuda is available"""
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            if have_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = audio_model(inputs)
            predictions = torch.argmax(outputs, 1)
            y_true.extend(labels.data.cpu())
            y_pred.extend(predictions.data.cpu())
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, normalize='pred')
    df_cm = pd.DataFrame(cf_matrix[start_index:, start_index:],
                        index = [i for i in classes_list[start_index:]],
                        columns = [i for i in classes_list[start_index:]])
    plt.figure(figsize = (20, 12) if start_index < 35 else (8, 4))
    sn.heatmap(df_cm, annot=True)
    plt.title('Tévesztési mátrix')