import resampy
import torchcrepe
import torch
import librosa
import numpy as np
import glob, os
from tqdm import tqdm

def get_pitch_crepe(wav_path, threshold=0.05):
    wav_data, _ = librosa.core.load(wav_path, sr=48000)
    mel_len = len(wav_data) // 256
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    # crepe只支持16khz采样率，需要重采样
    wav16k = resampy.resample(wav_data, 48000, 16000)
    wav16k_torch = torch.FloatTensor(wav16k).unsqueeze(0).to(device)

    # 频率范围
    f0_min = 50 #hparams['f0_min']
    f0_max = 1100.0 #hparams['f0_max']

    # 重采样后按照hopsize=80,也就是5ms一帧分析f0
    f0, pd = torchcrepe.predict(wav16k_torch, 16000, 80, f0_min, f0_max, pad=True, model='full', batch_size=1024,
                                device=device, return_periodicity=True)

    # 滤波，去掉静音，设置uv阈值，参考原仓库readme
    pd = torchcrepe.filter.median(pd, 3)
    pd = torchcrepe.threshold.Silence(-60.)(pd, wav16k_torch, 16000, 80)
    f0 = torchcrepe.threshold.At(threshold)(f0, pd)
    f0 = torchcrepe.filter.mean(f0, 3)

    # 将nan频率（uv部分）转换为0频率
    f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)

    # 去掉0频率，并线性插值
    nzindex = torch.nonzero(f0[0]).squeeze()
    f0 = torch.index_select(f0[0], dim=0, index=nzindex).cpu().numpy()
    time_org = 0.005 * nzindex.cpu().numpy()
    time_frame = np.arange(mel_len) * 256 / 48000

    if f0.shape[0] == 0:
        f0 = torch.FloatTensor(time_frame.shape[0]).fill_(0)
        print('f0 all zero!')
    else:
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])

    # pitch_coarse = f0_to_coarse(f0)
    return f0

if __name__ == "__main__":
    data_dir = "/data_disk/singing-data-preprocess-main/230227seg"
    for wav_path in tqdm(glob.glob(os.path.join(data_dir, "*", "*", "*.wav"))):
        f0 = get_pitch_crepe(wav_path)
        np.save(wav_path.replace(".wav", ".npy"), f0)