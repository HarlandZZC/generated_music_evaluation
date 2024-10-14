# Generated Music Evaluation

This is a repository designed to evaluate the effectiveness of AI-generated music, with extensibility for future metrics. Currently, the integrated evaluation metrics include:

* FAD_score: Frechet Audio Distance score(supporting frequency: any)
* KL_softmax: Kullback Leibler Divergence Softmax(AudioGen use this formulation) (supporting frequency: 16000, 32000)
* KL_sigmoid: Kullback Leibler Divergence Sigmoid(For multi-class audio clips, this formulation could be better) (supporting frequency: 16000, 32000)
* Clap_score: Contrastive Language-Audio Pretraining score(supporting frequency: 48000)

You can adjust the sampling rate of your audio by using the `--{xxx}sample_rate` parameter to resample the audio to different sampling rates. This allows you to evaluate the audio with various metrics at different resolutions, depending on the specific evaluation criteria you are using.

We drew inspiration from some of [Haohe Liu's code](https://github.com/haoheliu/audioldm_eval.git) and [Stability's code](https://github.com/Stability-AI/stable-audio-metrics.git). Many thanks!

To use the repo, you need to:

1. Clone the repository:

   ```bash
   git clone https://github.com/HarlandZZC/generated_music_evaluation.git
   cd generated_music_evaluation
   ```

2. Set up the environment:

   ```bash
   conda env create -f music_eval_env.yaml
   conda activate music_eval
   ```

   If some packages cannot be installed through the YAML file, please download them manually.

3. Adjust the `eval.sh` according to your need. You can also run `python eval.py --help` to see more information about the arguments.

4. Perform the evaluation:

    ```bash
    bash eval.sh
    ```

5. Check the terminal and your `output_path` set in `eval.sh` about the results.

6. Note:

   * To save time, when calculating FAD for the first time, we save the embeddings of the ref_data in `./temp_data/ref_data_embd_for_fad_{fad_sr}.npy`. This way, for subsequent FAD calculations, assuming the ref_data remains unchanged, we can directly load the values from `ref_data_embd_for_fad_{fad_sr}.npy`.
   * To save time, when calculating KL for the first time, we save the features of the ref_data in `./temp_data/ref_data_feat_for_kl_{kl_sr}.pkl`. This way, for subsequent KL calculations, assuming the ref_data remains unchanged, we can directly load the values from `ref_data_feat_for_kl_{kl_sr}.pkl`.
   * To save time, when calculating Clap for the first time, we save the features of the text in `./temp_data/text_embd_dict_for_clap.pth`. This way, for subsequent Clap calculations, assuming the text remains unchanged, we can directly load the values from `text_embd_dict_for_clap.pth`.
   * To improve readability, we set all values to display only up to 4 decimal places. If you wish to change this, please modify it in `eval.py`.
   * The audio files corresponding to `ref_path` and `gen_path` must have the same naming!
   * The file names in `ref_path` and `gen_path` only need to have matching numerical parts(but need to be unique). The code will automatically pair files with the same numerical part in their names and use this numerical part as their ID. For example, `123_ref.wav` in `ref_path` and `123_gen.wav` in `gen_path` will be considered corresponding files, with `123` serving as their common ID.
   * `--id2text_csv_path`should be organized like this:
  
      ```csv
      ids,descri
      "123","caption for 123."
      "456","caption for 456."
      ```
