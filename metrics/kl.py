import os
import pickle
import torch
import torchaudio
from tqdm import tqdm

def kl(ref_dataloader, gen_dataloader, origin_sr, kl_sr, device):
    # 1. build and set the model
    if kl_sr == 16000:
        mel_model = torch.hub.load("HarlandZZC/torchcnn14", "cnn14", 
                            device=device, features_list=["2048", "logits"], sample_rate = 16000, 
                            window_size=512, hop_size=160, mel_bins=64, fmin=50, fmax=8000, classes_num=527,
                            trust_repo=True)
    elif kl_sr == 32000:
        mel_model = torch.hub.load("HarlandZZC/torchcnn14", "cnn14", 
                            device=device, features_list=["2048", "logits"], sample_rate = 32000, 
                            window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527,
                            trust_repo=True)
    else:
        raise ValueError("Setted KL sample rate is not supported!")
    mel_model.eval()
    
    # 2. extract features for data
    def extract_feat(dataloader, model):
        out = None
        out_meta = None
        for waveform, filename in tqdm(dataloader):
            waveform = torchaudio.functional.resample(waveform, orig_freq=origin_sr, new_freq=kl_sr) 
            waveform = waveform - waveform.mean(dim=1, keepdim=True)
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
        return {**out, **out_meta}
    
    print("Extracting features for reference data...")
    if not os.path.exists("./temp_data"):
        os.makedirs("./temp_data")
    pkl_path = f"./temp_data/ref_data_feat_for_kl_{kl_sr}.pkl"
    if os.path.exists(pkl_path):
        print("Loading previous feature extraction results...")
        with open(pkl_path, "rb") as f:
            ref_feat_dict = pickle.load(f) 
    else:
        ref_feat_dict = extract_feat(ref_dataloader, mel_model)
        with open(pkl_path, "wb") as f:
             pickle.dump(ref_feat_dict, f)
    
    print("Extracting features for generated data...")
    gen_feat_dict = extract_feat(gen_dataloader, mel_model)

    # 3. Calculate KL divergence
    def calc_kl(feat_dict_1, feat_dict_2, feat_layer_name):
        EPS = 1e-6
        feat_1 = feat_dict_1[feat_layer_name]
        feat_2 = feat_dict_2[feat_layer_name]
        paths_1 = [os.path.basename(x) for x in feat_dict_1["file_path_"]]
        paths_2 = [os.path.basename(x) for x in feat_dict_2["file_path_"]]
        path_to_feats_1 = {p: f for p, f in zip(paths_1, feat_1)}
        path_to_feats_2 = {p: f for p, f in zip(paths_2, feat_2)}
        sharedkey_to_feats_1 = {p: path_to_feats_1[p] for p in paths_1}
        sharedkey_to_feats_2 = {p: path_to_feats_2[p] for p in paths_2}
        feats_1 = []
        feats_2 = []
        for sharedkey, feat_2 in sharedkey_to_feats_2.items():
            if sharedkey not in sharedkey_to_feats_1.keys():
                print("%s is not in the generation result" % sharedkey)
                continue
            feats_1.extend([sharedkey_to_feats_1[sharedkey]])
            feats_2.extend([feat_2])
        feats_1 = torch.stack(feats_1, dim=0)
        feats_2 = torch.stack(feats_2, dim=0)

        kl_ref = torch.nn.functional.kl_div(
            (feats_1.softmax(dim=1) + EPS).log(),
            feats_2.softmax(dim=1),
            reduction="none",
        ) / len(feats_1)
        kl_ref = torch.mean(kl_ref, dim=-1)

        # AudioGen use this formulation
        kl_softmax = torch.nn.functional.kl_div(
            (feats_1.softmax(dim=1) + EPS).log(),
            feats_2.softmax(dim=1),
            reduction="sum",
        ) / len(feats_1)

        # For multi-class audio clips, this formulation could be better
        kl_sigmoid = torch.nn.functional.kl_div(
            (feats_1.sigmoid() + EPS).log(), feats_2.sigmoid(), reduction="sum"
        ) / len(feats_1)   
    
        kl_softmax = kl_softmax.cpu().detach().numpy().item()
        kl_sigmoid = kl_sigmoid.cpu().detach().numpy().item()

        return kl_softmax, kl_sigmoid
    
    kl_softmax, kl_sigmoid = calc_kl(gen_feat_dict, ref_feat_dict, "logits")

    # 4. return KL divergence
    return kl_softmax, kl_sigmoid
