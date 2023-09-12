# LPR TROCR TH

train TrOCR script

## Main Package require

- python 3.8.10
- torch 2.0.1
- transformers 4.33.0

## How to run
### - install package
```bash
pip3 install -r requirements.txt
```
### - run train
```bash
python3 ./train.py
```

## Config
- root_dataset_dir = {dataset folder path}
    - dataset folder structure
        - th_train
- output_dir = model output path
- custom_model_name = "microsoft/trocr-<model name>"
- epochs
- learning_rate
- batch_size
