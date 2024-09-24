import argparse
import torch
from torch.utils.data import DataLoader
from useful_classes.dataset_template import dataset_template
from metrics.fad import fad

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_path", type=str, default="./reference")
    parser.add_argument("--generated_path", type=str, default="./generated")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_load_workers", type=int, default=8)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--run_fad", type=bool, default=True)

    args = parser.parse_args()
    reference_path = args.reference_path
    generated_path = args.generated_path
    output_path = args.output_path
    device_id = args.device_id
    batch_size = args.batch_size
    num_load_workers = args.num_load_workers
    sample_rate = args.sample_rate
    run_fad = args.run_fad

    # build dataset
    reference_dataset = dataset_template(reference_path, sample_rate)
    generated_dataset = dataset_template(generated_path, sample_rate)

    # build dataloader
    reference_dataloader = DataLoader(reference_dataset, batch_size=batch_size, 
                                      num_workers=num_load_workers, shuffle=False)
    generated_dataloader = DataLoader(generated_dataset, batch_size=batch_size, 
                                      num_workers=num_load_workers, shuffle=False)

    # set device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    if run_fad:
        fad(reference_dataloader, generated_dataloader, output_path, device)