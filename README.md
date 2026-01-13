<div align="center">    
 
# SRSC: Spatial and Relational Structural Constraints for Transformer-based Handwritten Mathematical Expression Recognition  

</div>

## Project structure
```bash
├── README.md
├── srsc               # model definition folder
├── convert2symLG       # official tool to convert latex to symLG format
├── lgeval              # official tool to compare symLGs in two folder
├── config.yaml         # config for SRSC hyperparameter
├── data.zip
├── eval_all.sh         # script to evaluate model on all CROHME test sets
├── requirements.txt
├── scripts             # evaluation scripts
└── train.py
```

## Install dependencies   
```bash
cd SRSC
# install project   
conda create -y -n SRSC python=3.9
conda activate SRSC
conda install -y pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -y pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge
conda install -y pandoc=2.19 -c conda-forge
pip install -r requirements.txt
 ```

## Training
Next, navigate to SRSC folder and run `train.py`. It may take **X** hours on **N** NVIDIA ABCXYZ gpus using ddp.
```bash
# Generate ground truth brefore training
python scripts/pregenerate_ground_truth.py --data_zip data.zip --output_dir data/cached_maps --split all
# Train SRSC model using N gpus and ddp
python train.py --config config.yaml  
```

You may change the `config.yaml` file to train different models
```yaml
Update later
```

For single gpu user, you may change the `config.yaml` file to
```yaml
gpus: 1
# gpus: 4
# accelerator: ddp
```

## Evaluation
Metrics used in validation during the training process is not accurate.

For accurate metrics reported in the paper, please use tools officially provided by CROHME 2019 oganizer:

A trained SRSC weight checkpoint has been saved in `lightning_logs/version_0` (update later)



```bash
perl --version  # make sure you have installed perl 5

unzip -q data.zip

# evaluation
# evaluate model in lightning_logs/version_0 on all CROHME test sets
# results will be printed in the screen and saved to lightning_logs/version_0 folder
bash eval_all.sh 0
```


