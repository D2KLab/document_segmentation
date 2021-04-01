"""
unfolding document structure ...
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import os
import gzip
import csv

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read IBM Debater extended")

train_samples = []
dev_samples = []
test_samples = []
with open('datasets/IBM_Debater_R_TCS_ACL_2018.v0/extended_dataset.csv', 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter=',')
    for row in reader:
        clazz = int(row['class'])
        if row['split'] == 'train':
            train_samples.append(InputExample(texts=[row['sentence_1'], row['sentence_2']], label=clazz))
        elif row['split'] == 'val':
            dev_samples.append(InputExample(texts=[row['sentence_1'], row['sentence_2']], label=clazz))
        elif row['split'] == 'test':
            test_samples.append(InputExample(texts=[row['sentence_1'], row['sentence_2']], label=clazz))

train_batch_size = 16
num_epochs = 100
#model_name = 'distilroberta-base' 
model_name = 'distilbert-base-uncased'
model_save_path = 'output/training_unfolding_structure-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'_'+model_name

#Define our CrossEncoder model. We use distilroberta-base as basis and setup it up to predict 3 labels
model = CrossEncoder(model_name, num_labels=1, device=3)

#We wrap train_samples, which is a list ot InputExample, in a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=False, batch_size=train_batch_size)

evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name='UnfoldingStructure-dev')

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=10000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

##############################################################################
# Testing
##############################################################################

logging.info("Read test examples")

model = CrossEncoder(model_save_path)
test_evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_samples, name='test')
test_evaluator(model, output_path=model_save_path)
