import torch
import numpy as np
import transformers
from transformers import BertTokenizer, BertModel, BertTokenizerFast, Trainer, \
    TrainingArguments, BertForSequenceClassification, BertConfig, BertForMaskedLM, \
    T5Model, T5Tokenizer, T5ForConditionalGeneration, T5Config, T5EncoderModel

import torch.nn as nn
import torch.nn.functional as F
import argparse

import os
from torch.utils.data import Dataset, DataLoader

from pymatgen.core.structure import Structure
from robocrys import StructureCondenser, StructureDescriber
import random
import pandas as pd
import glob

from tqdm import tqdm
from pathlib import Path

import warnings
from IPython import embed

import time

from utils import *

# Ignore all warnings
warnings.filterwarnings("ignore")

'''
class SimCLREncoder(nn.Module):
    def __init__(self, base_model="t5-base-uncased"):
        super(SimCLREncoder, self).__init__()

        self.bert = BertModel.from_pretrained(base_model)
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        representations = outputs.pooler_output
        return self.projection(representations)
'''
def evaluate(model, dataloader, device):
    model.eval()

    loss_sum = 0
    cnt = 0

    # save the representations of the structures
    z_i_list = []
    z_j_list = []
    
    nt_xent_criterion = NTXentLoss(batch_size=dataloader.batch_size, temperature=args.temperature)
    for batch in tqdm(dataloader):
        input_ids1, attention_mask1, input_ids2, attention_mask2, _ = batch

        with torch.no_grad():
            input_ids1, attention_mask1, input_ids2, attention_mask2 = \
                input_ids1.to(device), attention_mask1.to(device), input_ids2.to(device), attention_mask2.to(device)
            # embed()
            z_i, z_j = model(input_ids1, input_ids2, attention_mask1, attention_mask2)
            loss = nt_xent_criterion(z_i, z_j)

        loss_sum += loss.item()
        cnt += 1

        z_i_list.append(z_i)
        z_j_list.append(z_j)

    loss_avg = loss_sum / cnt

    # calculate the similarity between the representations
    z_i = torch.cat(z_i_list)
    z_j = torch.cat(z_j_list)
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    # similarity_matrix = F.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0), dim=2)
    # sim_ij = torch.diag(similarity_matrix)

    # calculate cosine similarity in chunks
    sim_ij = torch.tensor([]).to(device)
    chunk_size = 10
    for i in tqdm(range(0, len(z_i), chunk_size)):
        sim_ij = torch.cat([sim_ij, F.cosine_similarity(z_i[i:i+chunk_size].unsqueeze(1), z_j.unsqueeze(0), dim=2)], dim=0)

    # calculate the top 5 and top 1 accuracy
    top5_accuracy = 0
    top1_accuracy = 0
    rank_sum = 0.
    for i in range(len(sim_ij)):
        top5_indices = sim_ij[i].topk(5).indices
        top1_index = sim_ij[i].topk(1).indices
        # sort sim_ij[i] in descending order and take the sorted indices
        sorted_indices = torch.argsort(sim_ij[i], descending=True)
        rank = (sorted_indices == i).nonzero().item()

        if (top5_indices == i).any():
            top5_accuracy += 1

        if (top1_index == i).any(): 
            top1_accuracy += 1
        
        rank_sum += rank

    top5_accuracy /= len(sim_ij)
    top1_accuracy /= len(sim_ij)
    rank_avg = rank_sum / len(sim_ij)
    rank_ratio = rank_avg / len(sim_ij)

    metrics = {'loss': loss_avg, 'top5_accuracy': top5_accuracy, 'top1_accuracy': top1_accuracy, 'rank_avg': rank_avg, 'rank_ratio': rank_ratio}

    # release the memory
    del z_i_list, z_j_list, z_i, z_j, sim_ij
    del input_ids1, attention_mask1, input_ids2, attention_mask2
    # torch.cuda.empty_cache()

    model.train()
    return metrics

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", 
                        type=str, 
                        default="t5-base", 
                        help="Pretrained model to use as the base encoder.")

    parser.add_argument("--lr", 
                        type=float, 
                        default=1e-4, 
                        help="Learning rate for the optimizer.")

    parser.add_argument("--epochs", 
                        type=int, 
                        default=100, 
                        help="Number of epochs to train for.")

    parser.add_argument("--temperature", 
                        type=float, 
                        default=0.1, 
                        help="Temperature parameter for the NT-Xent loss.")

    parser.add_argument("--batch_size", 
                        type=int, 
                        default=32, 
                        help="Batch size for training.")
    
    parser.add_argument("--accumulate_grad_batches", 
                        type=int, 
                        default=8, 
                        help="Batch size for training.")

    parser.add_argument("--data_dir", 
                        type=Path, 
                        default="../data/basic",
                        help="Path to the directory containing training data.")
    parser.add_argument("--name",
                        type=str,
                        default="simclr_model",
                        help="Name of the model to save")
    parser.add_argument('--saved_dir',
                        type=str,
                        default="/data/rech/dingqian/intel/crystal-llm-retrieval",
                        help="Path to save the model")
    parser.add_argument('--evaluate_only',
                        action='store_true',
                        help="Whether to evaluate the model only")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()

    # fix random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # check if the csv with description is existed. If not, create one

    if not os.path.exists(args.data_dir / "train_desc.csv"):
        print("Creating description for the training data...")
        start_time = time.time()
        add_description_from_id(args.data_dir / "train.csv")
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

    # check saved_dir
    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)

    tokenizer = T5Tokenizer.from_pretrained(args.base_model)
    dataset = MyDataset(str(args.data_dir / "train_desc.csv"), tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader_eval = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simclr_model = SimCLR(tokenizer=dataset.tokenizer).to(device)
    simclr_model = nn.DataParallel(simclr_model)
    optimizer = torch.optim.AdamW(simclr_model.parameters(), lr=args.lr)
    nt_xent_criterion = NTXentLoss(batch_size=args.batch_size, temperature=args.temperature)

    if args.evaluate_only:
        simclr_model.load_state_dict(torch.load(os.path.join(args.saved_dir, f"{args.name}_best.pth")))
        metrics = evaluate(simclr_model, dataloader_eval, device)
        print(f"Metrics: {metrics}")
        exit()

    # set earlystopping
    num_bad_epochs = 0
    patience = 5
    best_loss = float("inf")
    print("Starting training...")
    for epoch in range(args.epochs):
        progress_bar = tqdm(dataloader)

        loss_sum = 0
        loss_avg = np.inf
        cnt = 0
        optimizer.zero_grad()

        for i, batch in enumerate(progress_bar):
            # embed()
            input_ids1, attention_mask1, input_ids2, attention_mask2, _ = batch
            input_ids1, attention_mask1, input_ids2, attention_mask2 = \
                input_ids1.to(device), attention_mask1.to(device), input_ids2.to(device), attention_mask2.to(device)
            
            z_i, z_j = simclr_model(input_ids1, input_ids2, attention_mask1, attention_mask2)
            loss = nt_xent_criterion(z_i, z_j)
            loss = loss / args.accumulate_grad_batches
            loss.backward()

            if (i + 1) % args.accumulate_grad_batches == 0 or i == len(dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()

                cnt += 1
                loss_avg = loss_sum / cnt

            loss_sum += loss.item()

            # Print loss value in progress bar
            # tqdm.write(f"Epoch {epoch+1}/{args.epochs} Loss: {loss.item()}")
            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs} Loss: {loss_avg}")

        loss_avg = loss_sum / cnt
        print(f"Epoch {epoch+1}/{args.epochs} Loss: {loss_avg}")

        # save loss values to log file, with epochs
        with open(f"log_{args.name}.txt", "a") as f:
            f.write(f"Epoch {epoch+1}/{args.epochs} Loss: {loss_avg}\n")
        
        # save model after each epoch
        if loss_avg < best_loss:
            best_loss = loss_avg
            saved_path = os.path.join(args.saved_dir, f"{args.name}_best.pth")
            torch.save(simclr_model.state_dict(), saved_path)
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1
            # delete the old model
        
        # metrics = evaluate(simclr_model, dataloader_eval, device)
        # print(f"Epoch {epoch+1}/{args.epochs} Metrics: {metrics}")

        if num_bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    saved_path = os.path.join(args.saved_dir, f"{args.name}_final.pth")
    torch.save(simclr_model.state_dict(), saved_path)
