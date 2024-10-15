python eval.py \
    --ref_path  ./data/stable-audio-metrics/groundtruth\
    --gen_path ./data/stable-audio-metrics/generated \
    --id2text_csv_path ./data/stable-audio-metrics/val.csv \
    --output_path ./output \
    --device_id 0 \
    --batch_size 32 \
    --original_sample_rate 24000 \
    --fad_sample_rate 16000 \
    --kl_sample_rate 16000 \
    --clap_sample_rate 48000 \
    --run_fad 1 \
    --run_kl 1 \
    --run_clap 1 \