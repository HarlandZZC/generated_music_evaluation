import os
import csv
import argparse
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from metrics.fad import fad
from metrics.kl import kl
from metrics.clap import clap
from utils.useful_classes import dataset_template

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_path", type=str, default="./data_example/ref", help="Path to the reference audio files")
    parser.add_argument("--gen_path", type=str, default="./data_example/gen", help="Path to the generated audio files")
    parser.add_argument("--id2text_csv_path", type=str, default="./data_example/id2text.csv", help="Path to the id2text csv file")
    parser.add_argument("--output_path", type=str, default="./output", help="Path to save the output files")
    parser.add_argument("--device_id", type=int, default=0, help="Device id to run the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--num_load_workers", type=int, default=8, help="Number of workers for dataloader")
    parser.add_argument("--original_sample_rate", type=int, default=16000, help="Sample rate of your original input audio files")
    parser.add_argument("--fad_sample_rate", type=int, default=16000, help="Sample rate for FAD")
    parser.add_argument("--kl_sample_rate", type=int, default=16000, help="Sample rate for KL")
    parser.add_argument("--clap_sample_rate", type=int, default=48000, help="Sample rate for CLAP")
    parser.add_argument("--run_fad", type=int, default=1, help="Run FAD(1) or not(0)")
    parser.add_argument("--run_kl", type=int, default=1, help="Run KL(1) or not(0)")
    parser.add_argument("--run_clap", type=int, default=1, help="Run CLAP(1) or not(0)")

    args = parser.parse_args()
    ref_path = args.ref_path
    gen_path = args.gen_path
    id2text_csv_path = args.id2text_csv_path
    output_path = args.output_path
    device_id = args.device_id
    batch_size = args.batch_size
    num_load_workers = args.num_load_workers
    origin_sr = args.original_sample_rate
    fad_sr = args.fad_sample_rate
    kl_sr = args.kl_sample_rate
    clap_sr = args.clap_sample_rate
    run_fad = args.run_fad
    run_kl = args.run_kl 
    run_clap = args.run_clap

    # 1. build data
    # 1.1 build dataset
    ref_dataset = dataset_template(ref_path)
    gen_dataset = dataset_template(gen_path)
    #1.2 build id2text
    id2text = {}
    with open(id2text_csv_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            id2text[row['ids']] = row['descri']

    # 2. build dataloader
    ref_dataloader = DataLoader(ref_dataset, batch_size=batch_size, 
                                      num_workers=num_load_workers, shuffle=False)
    gen_dataloader = DataLoader(gen_dataset, batch_size=batch_size, 
                                      num_workers=num_load_workers, shuffle=False)
    ref_idlist = getattr(ref_dataset, 'idlist', None)
    gen_idlist = getattr(gen_dataset, 'idlist', None)
    if ref_idlist == gen_idlist:
        print("ref_dataloader and gen_dataloader have the same idlist.")
    else:
        raise ValueError("ref_dataloader and gen_dataloader have different idlist.")

    # 3. set device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # 4. run metrics
    if run_fad == 1:
        print("--- Running FAD ---")
        fad_score = fad(ref_dataloader, gen_dataloader, origin_sr, fad_sr, device)
        fad_score = round(fad_score, 4)
        print(f"FAD Score: {fad_score}")
    
    if run_kl == 1:
        print("--- Running KL ---")
        kl_softmax, kl_sigmoid = kl(ref_dataloader, gen_dataloader, origin_sr, kl_sr, device)
        kl_softmax = round(kl_softmax, 4)
        kl_sigmoid = round(kl_sigmoid, 4)
        print(f"KL Softmax: {kl_softmax}")
        print(f"KL Sigmoid: {kl_sigmoid}")

    if run_clap == 1:
        print("--- Running CLAP ---")
        clap_score = clap(id2text, gen_dataloader, origin_sr, clap_sr, device)
        clap_score = round(clap_score, 4)
        print(f"CLAP Score: {clap_score}")

    # 5. save output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 5.1 add content to output
    output_list = [ref_path, gen_path]
    if run_fad == 1:
        output_list.append(fad_score)
    else:
        output_list.append("N/A")

    if run_kl == 1:
        output_list.append(kl_softmax)
        output_list.append(kl_sigmoid)
    else:
        output_list.append("N/A")
        output_list.append("N/A")

    if run_clap == 1:
        output_list.append(clap_score)
    else:
        output_list.append("N/A")

    # 5.2 write output to csv
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filepath = os.path.join(output_path, f"output_{current_time}.csv")
    with open(csv_filepath, mode="w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["ref_path", "gen_path", "FAD_score", "KL_softmax", "KL_sigmoid", "CLAP_score"])
        writer.writerow(output_list)