import os
import csv
import argparse
import torch
import torchaudio
from torch.utils.data import DataLoader
from datetime import datetime
from metrics.fad import fad

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_path", type=str, default="./data_example/ref", help="Path to the reference audio files")
    parser.add_argument("--gen_path", type=str, default="./data_example/gen", help="Path to the generated audio files")
    parser.add_argument("--output_path", type=str, default="./output", help="Path to save the output files")
    parser.add_argument("--device_id", type=int, default=0, help="Device id to run the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--num_load_workers", type=int, default=8, help="Number of workers for dataloader")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate for audio files")
    parser.add_argument("--run_fad", type=bool, default=True, help="Run FAD or not")

    args = parser.parse_args()
    ref_path = args.ref_path
    gen_path = args.gen_path
    output_path = args.output_path
    device_id = args.device_id
    batch_size = args.batch_size
    num_load_workers = args.num_load_workers
    sample_rate = args.sample_rate
    run_fad = args.run_fad

    # a template of dataset for building reference_dataset and generated_dataset
    class dataset_template(torch.utils.data.Dataset):
        def __init__(self, datadir, sr):
            self.datalist = [os.path.join(datadir, x) for x in os.listdir(datadir)]
            self.datalist = sorted(self.datalist)
            self.sr = sr

        def __getitem__(self, index):
            filename = self.datalist[index]
            waveform = self.read_from_file(filename)
            return waveform, os.path.basename(filename)

        def __len__(self):
            return len(self.datalist)
        
        def read_from_file(self, audio_file):
            audio, file_sr = torchaudio.load(audio_file)
            # resample if needed
            if file_sr != self.sr:
                audio = torchaudio.functional.resample(audio, orig_freq=file_sr, new_freq=self.sr)
            # only use the first channel    
            audio = audio.squeeze(0)
            # min-max normalization
            audio = audio - audio.mean()
            return audio

    # 1. build dataset
    ref_dataset = dataset_template(ref_path, sample_rate)
    gen_dataset = dataset_template(gen_path, sample_rate)

    # 2. build dataloader
    ref_dataloader = DataLoader(ref_dataset, batch_size=batch_size, 
                                      num_workers=num_load_workers, shuffle=False)
    gen_dataloader = DataLoader(gen_dataset, batch_size=batch_size, 
                                      num_workers=num_load_workers, shuffle=False)

    # 3. set device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # 4. run metrics
    if run_fad:
        fad_score = fad(ref_dataloader, gen_dataloader, device)
        print(f"FAD score: {fad_score}")

    # 5. save output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 5.1 add content to output
    output_list = [ref_path, gen_path]
    if run_fad:
        output_list.append(fad_score)
    else:
        output_list.append("N/A")

    # 5.2 write output to csv
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filepath = os.path.join(output_path, f"output_{current_time}.csv")
    with open(csv_filepath, mode="w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["ref_path", "gen_path", "FAD_score"])
        writer.writerow(output_list)