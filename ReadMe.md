## Pairwise similarity learning for confidence-based open-set viral taxonomy

#### Pre-requisites 

Install all the requirements from the requirements.txt file with the command:

```bash
pip install -r requirements.txt
```

Also, the dataset generation pipeline requires the installation of MMSeqs2. Additionally, currently it requires supplying a pickle file with HMM hits for the viral sequences. The pipeline will be extended to automate this step.

#### Dataset generation

To generate datsets, run the following command:

```bash
python -m scripts.generate_datasets --rank all --scenarios all --output-root <data_root>
```
To add protein cluster features, run the following command:

```bash
python -m scripts.generate_pc_features --rank all --scenarios all --output-root <data_root> --mmseqs-bin <path/to/mmseqs> 
```

Note that data generation is currently impossible, as the repository does not contain the required input files due to its size. Automatic public interface is one of the priorities of the future work.

#### Model training and evaluation

For training and evaluation of the model use this command:

```bash
python run_train_and_eval.py --config <config.yaml>
```

Default config is located in the root of the repository.