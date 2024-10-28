import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.fft as fft
from tqdm.auto import tqdm, trange
from sklearn.metrics import f1_score, balanced_accuracy_score
import numpy as np


def train_model(model,
                dataset,
                run_seed,
                eval_dataset=None,
                epochs=10,
                batch_size=128,
                device='cpu',
                lr=0.001,
                weights=None
                ):
    loss_fn = nn.CrossEntropyLoss(weights)
    model.to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    best_f1 = 0.84
    if eval_dataset:
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    for r in trange(epochs, desc="Epoch ...", leave=False):
        for sample, target in tqdm(dataloader, desc="Step ...", leave=False):
            sample = sample.to(device)
            target = target.to(device)
            logits = model(sample)
            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if eval_dataset:
            model.eval()
            true_labels = []
            pred_labels = []
            with torch.no_grad():
                for sample, target in tqdm(eval_dataloader, desc="Step ...", leave=False):
                    sample = sample.to(device)
                    logits = model(sample)
                    pred = logits.argmax(axis=1).cpu()
                    true_labels.append(target)
                    pred_labels.append(pred)
            true_labels = torch.concat(true_labels).numpy()
            pred_labels = torch.concat(pred_labels).numpy()
            f1 = f1_score(true_labels, pred_labels, average=None)
            f1_all = f1_score(true_labels, pred_labels, average='macro')
            print(f1)
            print('epoch ' + str(r) + ' ', f1_all)
            if f1_all > best_f1:
                torch.save(model, f"cnntrad_model_seed{run_seed}_ep{r+1}_tl{f1_all:.4f}.pt")
                best_f1 = f1_all
            model.train()


def evaluate_model(model,
                   dataset,
                   device='cpu'
                   ):
    model.to(device)
    model.eval()
    true_labels = []
    pred_labels = []
    dataloader = DataLoader(dataset, batch_size=5000)
    with torch.no_grad():
        for sample, target in tqdm(dataloader, desc="Step ...", leave=False):
            sample = sample.to(device)
            logits = model(sample)
            pred = logits.argmax(axis=1).cpu()
            true_labels.append(target)
            pred_labels.append(pred)
    true_labels = torch.concat(true_labels).numpy()
    pred_labels = torch.concat(pred_labels).numpy()
    f1 = f1_score(true_labels, pred_labels, average=None)
    f1_all = f1_score(true_labels, pred_labels, average='macro')
    print(f1)
    print(f1_all)