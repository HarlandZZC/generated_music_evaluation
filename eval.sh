python eval.py \
    --ref_path ./data_example/ref \
    --gen_path ./data_example/gen \
    --id2text_json_path ./data_example/id2text.json \
    --output_path ./output \
    --device_id 0 \
    --batch_size 32 \
    --original_sample_rate 16000 \
    --fad_sample_rate 16000 \
    --kl_sample_rate 32000 \
    --clap_sample_rate 48000 \
    --run_fad 1 \
    --run_kl 1 \
    --run_clap 1 \