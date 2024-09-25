# Generated Music Evaluation

This is a repository designed to evaluate the effectiveness of AI-generated music, with extensibility for future metrics. Currently, the integrated evaluation metrics include:

* Frechet Audio Distance(FAD).

We drew inspiration from some of [Haohe Liu's code](https://github.com/haoheliu/audioldm_eval.git). Many thanks!

To use the repo, you need to:

1. Clone the repository:

   ```bash
   git clone https://github.com/HarlandZZC/generated_music_evaluation.git
   cd generated_music_evaluation
   ```

2. Set up the environment:

   ```bash
   conda env create -f music_eval.yaml
   ```

   If some packages cannot be installed through the YAML file, please download them manually.

3. Adjust the eval.sh according to your need. You can also run `python eval.py --help` to see more information about the arguments.

4. Perform the evaluation:

    ```bash
    bash eval.sh
    ```

5. Check the `terminal` and `output_path` about the results.

6. Note:

   * To save time, when calculating FAD for the first time, we save the embeddings of the ref_data in `./temp_data/ref_data_embd_for_fad.npy`. This way, for subsequent FAD calculations, assuming the ref_data remains unchanged, we can directly load the values from `ref_data_embd_for_fad.npy`.
