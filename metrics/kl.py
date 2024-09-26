import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from tqdm import tqdm

def kl(ref_dataloader, gen_dataloader, device):
    # 1. build and set the model
    ref_sr = ref_dataloader.dataset.sr
    gen_sr = gen_dataloader.dataset.sr

    if ref_sr == gen_sr == 16000:
        mel_model = torch.hub.load("HarlandZZC/torchcnn14", "cnn14", 
                            device=device, features_list=["2048", "logits"], sample_rate = ref_sr, 
                            window_size=512, hop_size=160, mel_bins=64, fmin=50, fmax=8000, classes_num=527,
                            trust_repo=True)
    elif ref_sr == gen_sr == 32000:
        mel_model = torch.hub.load("HarlandZZC/torchcnn14", "cnn14", 
                            device=device, features_list=["2048", "logits"], sample_rate = ref_sr, 
                            window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527,
                            trust_repo=True)
    else:
        print("Sample rate not supported!")
    mel_model.eval()
    
    # 2. extract features for data
    def extract_feat(dataloader, model, device):
        out = None
        out_meta = None
        for waveform, filename in tqdm(dataloader):
            meta_dict = {"file_path_": filename,}
            with torch.no_grad():
                feat_dict = model(waveform)
            feat_dict = {k: [v.cpu()] for k, v in feat_dict.items()}
            if out is None:
                out = feat_dict
            else:
                out = {k: out[k] + feat_dict[k] for k in out.keys()}
            if out_meta is None:
                out_meta = meta_dict
            else:
                out_meta = {k: out_meta[k] + meta_dict[k] for k in out_meta.keys()}
        out = {k: torch.cat(v, dim=0) for k, v in out.items()} 
        from IPython import embed; embed(); os._exit(0)
        return {**out, **out_meta}
    
    print("Extracting features for reference data...")
    if not os.path.exists("./temp_data"):
        os.makedirs("./temp_data")
    pkl_path = "./temp_data/ref_data_feat_for_kl.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            ref_feat_dict = pickle.load(f)
    else:
        ref_feat_dict = extract_feat(ref_dataloader, mel_model, device)