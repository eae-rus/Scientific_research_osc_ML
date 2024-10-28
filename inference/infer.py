import torch
from torch.utils.data import DataLoader
from dataset import PQDDataset
from torch.optim import Adam
import torch.nn as nn
import torch.fft as fft
from tqdm.auto import tqdm, trange
from sklearn.metrics import f1_score, balanced_accuracy_score
import numpy as np
import pandas as pd
import time


if __name__ == "__main__":

    NUM_TESTS = 5

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    test_df = pd.read_csv('test.csv')
    test_dataset = PQDDataset(
                              df=test_df,
                              window_size=32,
                              target_mode=True
                            )

    model = torch.load("model.pt", map_location=device)
    model.to(device)
    model.eval()
    print(model)

    # speeds = []
    times = []
    dataloader = DataLoader(test_dataset, batch_size=1)
    with torch.no_grad():
        with tqdm(dataloader, unit=" batch", mininterval=3) as t:
            for i, (batch, _) in enumerate(t):

                batch = batch.to(device)

                start_time = time.time()
                output = model(batch)
                times.append(time.time() - start_time)

        # end_time = time.time()
        # print(f"Time: {end_time - start_time}")
        # speeds.append(len(test_dataloader) / (end_time - start_time))
        # print(f"Inference speed is {speeds[-1]} frames per second")

    # print(f"Average speed is {np.mean(speeds)} frames per second")
    print(
        f"Time of inference is min={min(times)*1000} ms, avg={np.mean(times)*1000} ms, max={max(times)*1000} ms per frame"
    )