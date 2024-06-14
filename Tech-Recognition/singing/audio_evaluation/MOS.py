import glob
import random
import os
import shutil
random.seed(1234)

def get_gtsamples(gt_paths, tgt_dir):
    songs2utts = {}
    for file in glob.glob(f"{gt_paths}/*.wav"):
        if "[G]" in file:
            basename = os.path.basename(file)
            songname = "#".join(basename.split("#")[:-1])
            if songname not in songs2utts:
                songs2utts[songname] = []
            songs2utts[songname].append(file)
    
    os.makedirs(tgt_dir, exist_ok=True)
    for key, value in songs2utts.items():
        print(key)
        samples = random.sample(value, 3)
        for sample in samples:
            shutil.copy(sample, tgt_dir)

def get_predsamples(wav_paths, tgt_dir, name):
    samples = []
    for file in glob.glob(f"{tgt_dir}/*.wav"):
        if "[G]" in file:
            samples.append(os.path.basename(file).replace("[G]", "[P]"))
    for sample in samples:
        pred_path = os.path.join(wav_paths, sample)
        target_path = pred_path.replace(wav_paths, tgt_dir)
        target_path = target_path.replace('.wav', f"_{name}.wav")
        shutil.copyfile(pred_path, target_path)


if __name__ == '__main__':
    gt_paths = "/home/renyi/hjz/NeuralSeq/checkpoints/staff/fs2_attn1/generated_160000_/wavs"
    tgt_dir = "/home/renyi/hjz/NeuralSeq/checkpoints/staff/mos"
    # get_gtsamples(gt_paths=gt_paths, tgt_dir=tgt_dir)

    # wav_paths = "/home/renyi/hjz/NeuralSeq/checkpoints/staff/fs2_attn1/generated_160000_/wavs"
    wav_paths = "/home/renyi/hjz/NeuralSeq/checkpoints/staff/postdiff/f0diffdur_preddur/wavs"
    get_predsamples(wav_paths, tgt_dir, "diff_f0diffdur")
