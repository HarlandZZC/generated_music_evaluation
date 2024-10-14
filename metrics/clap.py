import os
import torch
import torchaudio
from tqdm import tqdm

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
        text_embd_dict = torch.load(pth_path, map_location=device, weights_only=True)
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