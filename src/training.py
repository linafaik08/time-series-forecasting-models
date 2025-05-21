import pandas as pd
import numpy as np
from statsforecast.utils import ConformalIntervals

def split_train_test(df, horizon, column_id, column_date):
    test_frames = []
    train_frames = []
    for uid, group in df.groupby(column_id):
        group = group.sort_values(column_date)
        train, test = group[:-horizon], group[-horizon:]
        train_frames.append(train)
        test_frames.append(test)
    return pd.concat(train_frames), pd.concat(test_frames)

