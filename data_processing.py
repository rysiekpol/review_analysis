import gzip
import pandas as pd
import numpy as np
from pandas import DataFrame


def parse(filename):
    f = gzip.open(filename, 'r')
    entry = {}
    for vers in f:
        vers = vers.decode('utf8').strip()
        colon_pos = vers.find(":")
        if colon_pos == -1:
            yield entry
            entry = {}
            continue
        e_name = vers[:colon_pos]
        rest = vers[colon_pos + 2:]
        entry[e_name] = rest
    yield entry


def get_data() -> DataFrame:
    data = []
    for e in parse("Cell_Phones_&_Accessories.txt.gz"):
        data.append(e)

    df = pd.DataFrame(data[:-2])
    df["review/time"] = df["review/time"].astype(int)
    df["review/score"] = df["review/score"].astype(float)
    df = df.astype({col: str for col in df.columns if col not in ["review/time", "review/score"]})
    return df


# guarantee stability of test set
# count hash of each review
def test_set_check(identifier, test_ratio):
    return hash(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


# split data into train and test set
def split_train_test(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
