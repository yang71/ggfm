import torch
import argparse
from ggfm.data import args_print, renamed_load, random_walk_based_corpus_construction, construct_link_and_node
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForPreTraining

import warnings
warnings.filterwarnings("ignore")

import wandb
wandb.init(mode="disabled")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pre-training WalkLM on a given graph (heterogeneous / homogeneous)')
    parser.add_argument('--data_dir', type=str, default='/home/yjy/heteroPrompt/ggfm/ggfm/datasets/', help='The address of data for pretrain.')
    parser.add_argument('--pretrained_model_dir', type=str, default='/home/yjy/heteroPrompt/ggfm/ggfm/pretrained_model/walklm', help='The save address for the pretrained model.')
    parser.add_argument('--cuda', type=int, default=1, help='Avaiable GPU ID')
    parser.add_argument('--max_length', type=int, default=512, help='max_length')      


    args = parser.parse_args()
    args_print(args)

    if args.cuda != -1: device = torch.device("cuda:" + str(args.cuda))
    else: device = torch.device("cpu")

    # # random walk-based corpus construction
    # graph = renamed_load(open(args.data_dir+'graph.pk', 'rb'))
    # construct_link_and_node(graph, args.data_dir)
    # relations = ['publishes in conference', 'publishes in journal', 'publishes in repository', 'publishes in patent',
    #              'conference includes paper', 'journal includes paper', 'repository includes paper', 'patent includes paper',
    #              'cites paper', 'is cited by paper', 'has paper in field L0', 'has paper in field L3',
    #              'has paper in field L1', 'has paper in field L2', 'has paper in field L5', 'has paper in field L4',
    #              'is the last author of paper', 'is middle author of paper', 'is the first author of paper', 'in field',
    #              'belongs to field', 'in field L0', 'in field L3', 'in field L1',
    #              'in field L2', 'in field L5', 'in field L4', 'in affiliation',
    #              'has author', 'has last author', 'has middle author', 'has middle author']
    # random_walk_based_corpus_construction(args.data_dir, relations)

    datasets = load_dataset("text", data_files={"train": args.data_dir + 'rw_train_corpus.txt', \
                                                "validation": args.data_dir + 'rw_val_corpus.txt'})

    card = '/home/yjy/heteroPrompt/distilroberta-base'
    tokenizer = AutoTokenizer.from_pretrained(card, use_fast=True)
    model = AutoModelForPreTraining.from_pretrained(card)

    def tokenize_function(samples):
        return tokenizer(samples["text"], max_length=args.max_length, truncation=True)

    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=args.pretrained_model_dir + "test",
        overwrite_output_dir=False,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        dataloader_num_workers=8,
        load_best_model_at_end=True,
        gradient_accumulation_steps=20,
        num_train_epochs=6,
        learning_rate=0.0005,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.evaluate()

    trainer.save_model(args.pretrained_model_dir + "xyz")