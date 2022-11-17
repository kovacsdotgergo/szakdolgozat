import torch
import esc_dataset
import sys, os, subprocess
import wget

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
    """@brief   calculating mean and std for normalization on esc-50
    @param[in]  esc_path, n_fft, num_mel, hop_len, augment, normalize, log_mel, use_kaldi
                used for instantiation ESC_dataset
    @returns mean, std
    
    Documentation for calculating norm
    * kaldi, no augment, n_fft = 1024, hop_len = 256, n_mels = 128 -> mean = -6.773274898529053, std = 5.422067642211914
    * kaldi, no augment, n_fft = 1024, hop_len = 256, n_mels = 128, std_avg -> mean = -6.773274898529053, std = 3.796977996826172
    * waveform, no augment, n_fft = 1024, hop_len = 256, n_mels = 128 -> mean = -0.00014325266238301992, std = 0.12926168739795685
    * kaldi, no augment, n_fft = 1024, hop_len = 256, n_mels = 128, std_avg -> mean = -6.773274898529053, std = 3.796977996826172
    * waveform, no augment, n_fft = 1024, hop_len = 256, n_mels = 128, std_avg -> mean = -0.00014325266238301992, std = 0.09423764795064926
    * no kaldi, no augment, n_fft = 1024, hop_len = 256, n_mels = 128, std_avg -> mean = -51.28519821166992, std = 46.351680755615234
    """
    norm_dataset = esc_dataset.ESCdataset(esc_path, n_fft=n_fft, hop_length=hop_len, n_mels=num_mel, augment=augment,
                                normalize=normalize, log_mel=log_mel, use_kaldi=use_kaldi)
    mean, std = get_mean_and_std(norm_dataset, std_avg=False)
    return mean, std


def setup_env():
    if 'google.colab' in sys.modules:
        running_in = 'colab'
    elif os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None:
        running_in = 'kaggle'
    else:
        running_in = 'local'

    #checking required repositories
    if running_in=='local':
        if not os.path.exists('./ast'):
            print('Clone AST from https://github.com/YuanGongND/ast')
            raise FileNotFoundError('AST repository not found')        
        if not os.path.exists('./ESC-50'):
            print('Clone ESC-50 from https://github.com/karolpiczak/ESC-50.git')
            raise FileNotFoundError('ESC-50 repository not found')
    else:
        if not os.path.exists('./ast'):
            subprocess.run('git clone https://github.com/YuanGongND/ast')
        if not os.path.exists('./ESC-50'):
            subprocess.run('git clone https://github.com/karolpiczak/ESC-50.git')
    if os.getcwd() + '/ast' not in sys.path:
        sys.path.append(os.getcwd() + '/ast')

    #for saving the models when training
    save_path = os.getcwd() + '/saved'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    #for the pretrained model
    #TODO torch_home
    pretrained_path = os.getcwd() + '/ast/pretrained_models'
    audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
    pretrained_model_name = 'audioset_10_10_0.4593.pth'
    if not os.path.exists(pretrained_path + '/' + pretrained_model_name):
        wget.download(url=audioset_mdl_url, out=pretrained_path + '/' + pretrained_model_name)

    #change working dir for ast, to find pretrained model
    #TODO: first run in function
    os.chdir(os.getcwd() + '/ast/src/models')
        #TODO from wsl_kickin, try out the unified notebook in all envs