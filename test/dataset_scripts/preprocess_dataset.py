import numpy as np
import pandas as pd


def transform_poker():
    data_path = 'data/datasets/poker.txt'
    saved_path = 'data/datasets/poker.csv'
    data = list()
    with open(data_path, 'r') as f:
        for line in f.readlines()[1:10001]:
            sample = [0.] * 11
            fields = line.strip()[1:-1].split(',')
            for item in fields:
                idx, val = item.strip().split(' ')
                idx = int(idx)
                if idx == 0:
                    sample[10] = int(val)
                else:
                    sample[idx-1] = float(val)
            assert sample[-1] in [-1, 1]
            data.append(sample)
    pd.DataFrame(np.asarray(data)).to_csv(saved_path, index=False)


if __name__ == "__main__":
    transform_poker()
