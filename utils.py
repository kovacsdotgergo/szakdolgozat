import torch
import dataset

def get_mean_and_std(dataset, std_avg=True):
    """
    @brief calculates the mean and standard deviation on the data
    @returns      mean, std
    @param[in]    dataset   dataset to calculate mean on
    @param[in]    std_avg      bool, true if std is average of stds, else std of all data
    """
    alldata = []
    stds = []
    for data, _ in dataset:
        alldata.append(data)
        if std_avg:
            stds.append(torch.std(data))
    #stacking all the data
    alldata = torch.stack(alldata, 0)
    #mean and std on the concatenated data
    mean = torch.mean(alldata)
    if std_avg:
        std = torch.mean(torch.tensor(stds))
    else:
        std = torch.std(alldata)
    return mean, std

def calculate_esc_mean_and_std(esc_path, n_fft=1024, num_mel=128, hop_len=256, augment=False,
                                normalize=False, log_mel=True, use_kaldi=True):
    norm_dataset = dataset.ESCdataset(esc_path, n_fft=n_fft, hop_length=hop_len, n_mels=num_mel, augment=augment,
                                normalize=normalize, log_mel=log_mel, use_kaldi=use_kaldi)
    mean, std = get_mean_and_std(norm_dataset)
    print(f'mean: {mean}, std: {std}')
    return mean, std
    #TODO try out

def setup_env():
    #TODO from wsl_kickin, try out the unified notebook in all envs