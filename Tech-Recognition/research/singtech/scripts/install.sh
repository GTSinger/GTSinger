conda create -n tech-recog python==3.9
conda activate tech-recog
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow==2.9.0 tensorflow-estimator==2.9.0 tensorboardX==2.5
pip install pyyaml matplotlib==3.5 pandas pyworld==0.2.12 librosa torchmetrics
pip install mir_eval pretty_midi pyloudnorm scikit-image textgrid g2p_en npy_append_array einops webrtcvad
export PYTHONPATH=.