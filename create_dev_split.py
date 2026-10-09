import os

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 11711
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

train_df = pd.read_csv(os.path.join(DATA, "etpc-paraphrase-train.csv"))

train_df, dev_df = train_test_split(
    train_df,
    test_size=0.2,
    random_state=SEED,
    shuffle=True
)

train_df.to_csv(
    os.path.join(DATA, "etpc-paraphrase-train-split.csv"),
    index=False
)

dev_df.to_csv(
    os.path.join(DATA, "etpc-paraphrase-dev.csv"),
    index=False
)

print(f"Train size: {len(train_df)}")
print(f"Dev size: {len(dev_df)}")
