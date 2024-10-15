import torch
import os
import numpy as np
import torchaudio
from scipy import linalg
from tqdm import tqdm
from torch import nn

def fad(ref_dataloader, gen_dataloader, origin_sr, fad_sr, device):
    # 1. build and set the model
    vggish_model = torch.hub.load("HarlandZZC/torchvggish", "vggish", 
                           device=device, postprocess = False, pretrained=True, preprocess=True, progress=True,
                           trust_repo=True)
    vggish_model.embeddings = nn.Sequential(*list(vggish_model.embeddings.children())[:-1])
    vggish_model.eval()

    # 2. load data to RAM
    def load_data_to_ram(dataloader):
        data_list = []
        for batch in tqdm(dataloader): 
            batch[0] = torchaudio.functional.resample(batch[0], orig_freq=origin_sr, new_freq=fad_sr)
            batch[0] = batch[0] - batch[0].mean(dim=1, keepdim=True) 
            for i in range(len(batch[0])): 
                data_list.append((batch[0][i])) 
        return data_list
    
    print("Loading reference data to RAM...")
    ref_data_list = load_data_to_ram(ref_dataloader)
    print("Loading generated data to RAM...")
    gen_data_list = load_data_to_ram(gen_dataloader)  
    
    # 3. extract embeddings for data
    def extract_embd(model, data_list, sr):
        embd_list = []
        for audio in tqdm(data_list):
            with torch.no_grad():
                embd = model.forward(audio.numpy(), sr)
            if model.device.type == "cuda":
                embd = embd.cpu()
            embd = embd.detach().numpy() 
            embd_list.append(embd)
        embd_array = np.concatenate(embd_list, axis=0)
        return embd_array

    print("Extracting embeddings for reference data...")
    if not os.path.exists("./temp_data"):
        os.makedirs("./temp_data")
    npy_path = f"./temp_data/ref_data_embd_for_fad_{fad_sr}.npy"
    if os.path.exists(npy_path):
        print("Loading previous embeddings extraction results...")
        ref_embd_array = np.load(npy_path)
    else:
        ref_embd_array = extract_embd(vggish_model, ref_data_list, fad_sr)
        np.save(npy_path, ref_embd_array)

    print("Extracting embeddings for generated data...")
    gen_embd_array = extract_embd(vggish_model, gen_data_list, fad_sr)

    # 4. calculate embedding statistics
    def calc_embd_stats(embd_array):
        mu = np.mean(embd_array, axis=0)
        sigma = np.cov(embd_array, rowvar=False)
        return mu, sigma
    
    ref_mu, ref_sigma = calc_embd_stats(ref_embd_array)
    gen_mu, gen_sigma = calc_embd_stats(gen_embd_array) 
    
    # 5. calculate FAD score
    # adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    def calc_fad(mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2) 

        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"
        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    fad_score = calc_fad(gen_mu, gen_sigma, ref_mu, ref_sigma) 

    # 6. return FAD score
    return fad_score