import os
import pickle
import torch
import torchaudio
import pyloudnorm as pyln
from tqdm import tqdm
from utils.useful_defs import int16_to_float32, float32_to_int16, load_state_dict

"""
Cosine similarity is computed between the LAION-CLAP text embedding of the given prompt and 
the LAION-CLAP audio embedding of the generated audio. LION-CLAP: https://github.com/LAION-AI/CLAP

This evaluation script assumes that audio_path files are identified with the ids in id2text.

clap_score() evaluates all ids in id2text.

GPU-based computation.

Select one of the following models from https://github.com/LAION-AI/CLAP:
    - music_speech_audioset_epoch_15_esc_89.98.pt (used by musicgen)
    - music_audioset_epoch_15_esc_90.14.pt
    - music_speech_epoch_15_esc_89.25.pt
    - 630k-audioset-fusion-best.pt (our default, with "fusion" to handle longer inputs)

Params:
-- id2text: dictionary with the mapping between id (generated audio filenames in audio_path) 
            and text (prompt used to generate audio). clap_score() evaluates all ids in id2text.
-- audio_path: path where the generated audio files to evaluate are available.
-- audio_files_extension: files extension (default .wav) in eval_path.
-- clap_model: choose one of the above clap_models (default: '630k-audioset-fusion-best.pt').
Returns:
-- CLAP-LION score
"""

def clap(id2text, gen_dataloader, origin_sr, clap_sr, device):
    # 1. build and set the model
    if clap_sr == 48000:
        clap_model = torch.hub.load("HarlandZZC/torchclapmodel", "clapmodel", 
                                weights_name="music_speech_audioset_epoch_15_esc_89.98.pt", device=device,
                                trust_repo=True)
    else:
        raise ValueError("Setted CLAP sample rate is not supported!")
    clap_model.eval()

    # 2. Extract embeddings for text
    print('Extract embeddings for text...')
    if not os.path.exists("./temp_data"):
        os.makedirs("./temp_data")
    pth_path = f"./temp_data/text_embd_dict_for_clap.pth"
    if os.path.exists(pth_path):
        print("Loading previous text embeddings extraction results...")
        text_embd_dict = torch.load(pth_path)
        text_embd_dict = {key: emb.to(device) for key, emb in text_embd_dict.items()}
    else:
        batch_size = gen_dataloader.batch_size
        text_embd_dict = {}
        for i in tqdm(range(0, len(id2text), batch_size)):
            batch_ids = list(id2text.keys())[i:i+batch_size]
            batch_texts = [id2text[id] for id in batch_ids] 
            with torch.no_grad(): 
                embd = clap_model.get_text_embedding(batch_texts, use_tensor=True) 
            for id, emb in zip(batch_ids, embd):
                text_embd_dict[id] = emb
        torch.save(text_embd_dict, pth_path)

    # 3. Extract embeddings for audio and calculate cosine similarity
    score = 0
    count = 0
    print('Extract embeddings for audio...')
    for audio, filename in tqdm(gen_dataloader):
        with torch.no_grad(): 
            audio = torchaudio.functional.resample(audio, orig_freq=origin_sr, new_freq=clap_sr) 
            max_val = torch.max(torch.abs(audio), dim=1, keepdim=True).values
            audio = audio / max_val 
            # audio = (audio * 32767).short() 
            # audio = audio.float() / 32767 
            audio_embd = clap_model.get_audio_embedding_from_data(x = audio, use_tensor=True) 
            text_embd = torch.stack([text_embd_dict[file] for file in filename], dim=0) 
        
            cosine_sim = torch.nn.functional.cosine_similarity(audio_embd, text_embd, dim=1, eps=1e-8)
            score += cosine_sim.sum()
            count += cosine_sim.size(0)
    
    # Calculate clap score
    clap_score = score / count if count > 0 else 0 
    clap_score = clap_score.cpu().detach().numpy().item() 
    return clap_score