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
            logits = torch.squeeze(logits)
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
                    logits = torch.squeeze(logits)
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
                torch.save(model, f"rknngru_model_seed{run_seed}_ep{r+1}_tl{f1_all:.4f}.pt")
                best_f1 = f1_all
            model.train()


def train_npu_model(model,
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
            sample = sample.permute(0, 2, 1)
            sample = sample.unsqueeze(2)
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
                    sample = sample.permute(0, 2, 1)
                    sample = sample.unsqueeze(2)
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
                torch.save(model, f"cnnt_npu_model_seed{run_seed}_ep{r+1}_tl{f1_all:.4f}.pt")
                best_f1 = f1_all
            model.train()


class SpectralLoss(nn.Module):
    def __init__(self, weights=None):
        super(SpectralLoss, self).__init__()
        self.weights = weights

    def forward(self, input_signal, target_signal, targets=None):
        input_spectrum = torch.abs(fft.rfft(input_signal))
        target_spectrum = torch.abs(fft.rfft(target_signal))

        # если заданы веса классов
        if self.weights is not None and targets is not None:
            # losses = torch.mean(
            #     torch.abs(input_spectrum - target_spectrum), axis=[1, 2]
            # )
            # стратегия выбора таргета для каждого сэмпла - выбрать просто
            # максимальное значение в смысле важности события:
            # targets = torch.max(targets, axis=1).values
            # weights = torch.empty(targets.shape, device=targets.device)
            # for i, target in enumerate(targets):
            #     weights[i] = self.weights[target.item()]

            # loss = torch.mean(losses * weights)
            assert False, "not implemented"
        else:
            loss = torch.mean(torch.abs(input_spectrum - target_spectrum))

        return loss


def train_ae_model(model,
                  dataset,
                  run_seed,
                  eval_dataset=None,
                  epochs=10,
                  batch_size=128,
                  device='cpu',
                  lr=0.001,
                  weights=None
                  ):
    loss_fn1 = nn.CrossEntropyLoss(weights)
    loss_fn2 = SpectralLoss()
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
            logits, r_sample = model(sample)
            loss1 = loss_fn1(logits, target)
            loss2 = loss_fn2(r_sample, sample)
            coef = (loss1 / loss2).detach()
            loss = loss1 + coef * loss2
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
                    logits, r_sample = model(sample)
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
                torch.save(model, f"cnnae_model_seed{run_seed}_ep{r+1}_tl{f1_all:.4f}.pt")
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
            logits = torch.squeeze(logits)
            pred = logits.argmax(axis=1).cpu()
            true_labels.append(target)
            pred_labels.append(pred)
    true_labels = torch.concat(true_labels).numpy()
    pred_labels = torch.concat(pred_labels).numpy()
    f1 = f1_score(true_labels, pred_labels, average=None)
    f1_all = f1_score(true_labels, pred_labels, average='macro')
    print(f1)
    print(f1_all)


def evaluate_npu_model(model,
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
            sample = sample.permute(0, 2, 1)
            sample = sample.unsqueeze(2)
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


def evaluate_ae_model(model,
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
            logits, r_sample = model(sample)
            pred = logits.argmax(axis=1).cpu()
            true_labels.append(target)
            pred_labels.append(pred)
    true_labels = torch.concat(true_labels).numpy()
    pred_labels = torch.concat(pred_labels).numpy()
    f1 = f1_score(true_labels, pred_labels, average=None)
    f1_all = f1_score(true_labels, pred_labels, average='macro')
    print(f1)
    print(f1_all)


class SpectralLoss2(nn.Module):
    def __init__(self, weights=None):
        super(SpectralLoss2, self).__init__()
        self.weights = weights

    def forward(self, input_signal, target_signal, targets=None):
        target_spectrum = torch.abs(fft.fft(target_signal))[:,1:16,:]
        target_spectrum, _ = torch.max(target_spectrum, dim=1)
        #print(target_spectrum.shape)
        #print(input_signal.shape)

        # если заданы веса классов
        if self.weights is not None and targets is not None:
            # losses = torch.mean(
            #     torch.abs(input_spectrum - target_spectrum), axis=[1, 2]
            # )
            # стратегия выбора таргета для каждого сэмпла - выбрать просто
            # максимальное значение в смысле важности события:
            # targets = torch.max(targets, axis=1).values
            # weights = torch.empty(targets.shape, device=targets.device)
            # for i, target in enumerate(targets):
            #     weights[i] = self.weights[target.item()]

            # loss = torch.mean(losses * weights)
            assert False, "not implemented"
        else:
            loss = torch.mean(torch.abs(input_signal - target_spectrum))

        return loss


def train_fft_model(model,
                dataset,
                run_seed,
                eval_dataset=None,
                epochs=10,
                batch_size=128,
                device='cpu',
                lr=0.001,
                weights=None
                ):
    loss_fn1 = nn.CrossEntropyLoss(weights)
    loss_fn2 = SpectralLoss2()
    model.to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    best_f1 = 0.84
    if eval_dataset:
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    for r in trange(epochs, desc="Epoch ...", leave=False):
        l1 = []
        l2 = []
        for sample, target in tqdm(dataloader, desc="Step ...", leave=False):
            sample = sample.to(device)
            target = target.to(device)
            logits, x_fft = model(sample)
            loss1 = loss_fn1(logits, target)
            loss2 = loss_fn2(x_fft, sample)
            l1.append(loss1.item())
            l2.append(loss2.item())
            loss = loss1 + 0.0001 * loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(np.mean(l1))
        print(np.mean(l2))
        if eval_dataset:
            model.eval()
            true_labels = []
            pred_labels = []
            with torch.no_grad():
                for sample, target in tqdm(eval_dataloader, desc="Step ...", leave=False):
                    sample = sample.to(device)
                    logits, x_fft = model(sample)
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
                torch.save(model, f"cnn_ftt_model_seed{run_seed}_ep{r+1}_tl{f1_all:.4f}.pt")
                best_f1 = f1_all
            model.train()