import os
n_cores = str(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = n_cores
os.environ['MKL_NUM_THREADS'] = n_cores
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import datasets
import polars as pl
import numpy as np

from sklearn.model_selection import train_test_split

### PARQUE DATASET ###

columns = ['sentiment', 'wav2numpy']
df = pl.read_parquet('sqe_messai.parquet', columns = columns)
#df = pl.read_parquet('sqe_messai.parquet', columns = columns, n_rows = 100)

### LABELS ###

slist = sorted(df['sentiment'].value_counts()['sentiment'].to_list())
minv = slist[0] * -1
reslist = [ int(round((i + minv) * 10, 0)) for i in slist ]

slist = [str(i) for i in slist]

label2id = dict(zip(slist, reslist))
id2label = dict(zip(reslist, slist))

### SPLIT DATASET ###

df = df.rename({'sentiment': 'label'})
df = df.with_columns(pl.col('label').cast(pl.Utf8))

train_df, test_df = train_test_split(df, test_size=0.2, random_state=31416)
train_dataset = datasets.Dataset.from_pandas(train_df.to_pandas())
test_dataset = datasets.Dataset.from_pandas(test_df.to_pandas())

dataset = datasets.DatasetDict({'train': train_dataset, 'test': test_dataset})
dataset = dataset.class_encode_column('label')

### AUDIO PREPROCESSING AND EMBEDDINGS EXTRACTION ###

from transformers import AutoFeatureExtractor
feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base')

def preprocess_function(examples):
	#inputs = feature_extractor(examples['wav2numpy'], sampling_rate=16000, padding=True)
	inputs = feature_extractor(examples['wav2numpy'], sampling_rate=16000, padding=True, max_length=480000, truncation=True)
	return inputs

encoded_dataset = dataset.map(preprocess_function, remove_columns='wav2numpy', batched=True)

### EVALUATION FUNCTION ###

import evaluate

accuracy = evaluate.load('accuracy')

def compute_metrics(eval_pred):
	predictions = np.argmax(eval_pred.predictions, axis=1)
	return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
	
### TRAIN ###

from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

num_labels = len(id2label)

model = AutoModelForAudioClassification.from_pretrained('facebook/wav2vec2-base', num_labels=num_labels, label2id=label2id, id2label=id2label)

training_args = TrainingArguments(
	output_dir='my_test_model',
	evaluation_strategy='epoch',
	save_strategy='epoch',
	learning_rate=2e-5,
	weight_decay=0.01,
	gradient_checkpointing=True,
	per_device_train_batch_size=8,
	#gradient_accumulation_steps=1,
	per_device_eval_batch_size=8,
	eval_accumulation_steps=1,
	num_train_epochs=10,
	warmup_ratio=0.1,
	#logging_steps=10,
	eval_steps=10,
	load_best_model_at_end=True,
	metric_for_best_model='accuracy',
	overwrite_output_dir=True,
	greater_is_better=True,
	optim='adamw_torch',
	push_to_hub=False,
	do_train=True,
	do_eval=True,
	do_predict=True,
)

trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=encoded_dataset['train'],
	eval_dataset=encoded_dataset['test'],
	tokenizer=feature_extractor,
	compute_metrics=compute_metrics,
)

trainer.train()
