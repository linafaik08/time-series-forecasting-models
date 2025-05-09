import pandas as pd
import numpy as np

def split_train_test(df, h):
    test_frames = []
    train_frames = []
    for uid, group in df.groupby("id"):
        group = group.sort_values("date")
        train, test = group[:-h], group[-h:]
        train_frames.append(train)
        test_frames.append(test)
    return pd.concat(train_frames), pd.concat(test_frames)