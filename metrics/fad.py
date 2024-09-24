import torch
import os
from tqdm import tqdm
from torch import nn

def fad(reference_dataloader, generated_dataloader, output_path, device):
    # build model
    model = torch.hub.load("harritaylor/torchvggish", "vggish")
    model.postprocess = False
    model.embeddings = nn.Sequential(
                *list(model.embeddings.children())[:-1]
            )
    model.to(device)
    model.eval()

    # extract embeddings
    # optimize the code later to increase the speed
    data_list = []
    print("Loading data to RAM")
    for batch in tqdm(generated_dataloader):
        for i in range(len(batch[0])):
            data_list.append((batch[0][i], 16000))
    from IPython import embed; embed(); os._exit(0)
    return data_list