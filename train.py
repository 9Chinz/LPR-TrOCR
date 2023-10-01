# config
root_dataset_dir = "dataset_2/"
output_dir = "./output_model/out-str-test-2"
#custom_model_name = "microsoft/trocr-base-printed"
custom_model_name = "microsoft/trocr-base-str"
epochs = 100 # 10
learning_rate = 5e-5 # 5e-5
batch_size = 16 # 4
n_workers = 32
eval_round = 25
# end config

import torch
import os
from accelerate import Accelerator

os.environ["TOKENIZERS_PARALLELISM"] = "true"

print(f"{'='*20} Check cuda availability {'='*20}")

print(torch.cuda.is_available())

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()

    if num_gpus > 0:
        for gpu in range(num_gpus):
            print(f"GPU {gpu}: {torch.cuda.get_device_name(gpu)}")
    else:
        print("No gpu")
else:
    print("cuda not available")
    
print(f"{'='*20} end check cuda availability {'='*20}")

print(f"{'='*20} Prepare dataset {'='*20}")

import pandas as pd
df_dataset1 = pd.read_csv(f'./dataset/{root_dataset_dir}th_train/labels.csv', encoding='utf8')
df_dataset1.rename(columns={"filename": "file_name", "words": "text"}, inplace=True)
print(df_dataset1.columns)
print(df_dataset1.head())
print(len(df_dataset1))

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df_dataset1, test_size=0.05)
# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
print(train_df)

import torch
from torch.utils.data import Dataset
from PIL import Image

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
    

from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained(custom_model_name)

train_dataset = IAMDataset(root_dir=f'./dataset/{root_dataset_dir}th_train/',
                           df=train_df,
                           processor=processor)
eval_dataset = IAMDataset(root_dir=f'./dataset/{root_dataset_dir}th_train/',
                           df=test_df,
                           processor=processor)

print(f"{'*'*20} result overview dataset {'*'*20}")
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))

encoding = train_dataset[0]
for k,v in encoding.items():
    print(k, v.shape)
    
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=n_workers)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=n_workers)

print(f"{'='*20} End prepare dataset {'='*20}")
print(f"{'='*20} Start training {'='*20}")

from transformers import VisionEncoderDecoderModel

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# add accelerator
accelerator = Accelerator(gradient_accumulation_steps=4)
device = accelerator.device
print(f'{"="*20} Using device: {device} {"="*20}')

model = VisionEncoderDecoderModel.from_pretrained(custom_model_name)
model.to(device)

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4
model.config.use_cache = True

import evaluate

cer_metric = evaluate.load("cer")

def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer

print("In training loop... ")

res_df = pd.DataFrame(columns=['epoch', 'train_loss'])

from tqdm import tqdm

import time

t1 = time.time()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
least_cer = 1.0
best_epoch = 0

# save model
os.makedirs(output_dir, exist_ok=True)

# accelerate prepare
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

for epoch in range(epochs):  # loop over the dataset multiple times
    # train
    t1_train = time.time()
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_dataloader):
        with accelerator.accumulate(model):
            # get the inputs
            for k,v in batch.items():
                batch[k] = v.to(device)

            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            # using accelerate
            accelerator.backward(loss)
            #loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

    print(f"Epoch {epoch} took {time.time() - t1_train} seconds")
    print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
    
    # evaluate
    if ((epoch+1) % eval_round == 0):
        eval_t1 = time.time()
        model.eval()
        valid_cer = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                # run batch generation
                outputs = model.generate(batch["pixel_values"].to(device),)
                # compute metrics
                cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
                valid_cer += cer 
        print(f"Evaluation took {time.time() - eval_t1} seconds")
        print("Validation CER:", valid_cer / len(eval_dataloader))
    
        # save model if it has the best cer
        if valid_cer / len(eval_dataloader) < least_cer:
            print(f"{'!'*20} Saving model with lowest CER {'!'*20}")
            best_epoch = epoch
            least_cer = valid_cer / len(eval_dataloader)
            model.save_pretrained(output_dir)
        
    res_df.loc[epoch] = {'epoch': epoch+1, 'train_loss': train_loss/len(train_dataloader)}

print(f"{'='*20} Finished Training {'='*20}")

# save last epcoh model
sub_dir = f"{output_dir}/checkpoint-{epochs}"
os.makedirs(sub_dir, exist_ok=True)
model.save_pretrained(sub_dir)

print(f"{'='*20} total time trained {'='*20}")
from datetime import timedelta

def get_time_hh_mm_ss(sec):
    # create timedelta and convert it into string
    td_str = str(timedelta(seconds=sec))
    print('Time in seconds:', sec)

    # split string into individual component
    x = td_str.split(':')
    return f'Time in hh:mm:ss: {x[0]} Hours {x[1]} Minutes {x[2]} Seconds'

print(get_time_hh_mm_ss(time.time() - t1))

print(f"{'='*20} train loss {'='*20}")

from tabulate import tabulate

print(tabulate(res_df, headers='keys', tablefmt='psql'))
print("Best epoch:", best_epoch+1)

