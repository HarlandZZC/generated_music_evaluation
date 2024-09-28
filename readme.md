# Generated Music Evaluation

This is a repository designed to evaluate the effectiveness of AI-generated music, with extensibility for future metrics. Currently, the integrated evaluation metrics include:

* FAD_score: Frechet Audio Distance score
* KL_softmax: Kullback Leibler Divergence Softmax(AudioGen use this formulation)
* KL_sigmoid: Kullback Leibler Divergence Sigmoid(For multi-class audio clips, this formulation could be better)

We drew inspiration from some of [Haohe Liu's code](https://github.com/haoheliu/audioldm_eval.git). Many thanks!

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

   * To save time, when calculating FAD for the first time, we save the embeddings of the ref_data in `./temp_data/ref_data_embd_for_fad.npy`. This way, for subsequent FAD calculations, assuming the ref_data remains unchanged, we can directly load the values from `ref_data_embd_for_fad.npy`.
   * To save time, when calculating KL for the first time, we save the features of the ref_data in `./temp_data/ref_data_feat_for_kl.pkl`. This way, for subsequent KL calculations, assuming the ref_data remains unchanged, we can directly load the values from `ref_data_feat_for_kl.pkl`.
   * To improve readability, we set all values to display only up to 4 decimal places. If you wish to change this, please modify it in `eval.py`.
   * The audio files corresponding to `ref_path` and `gen_path` must have the same naming!
