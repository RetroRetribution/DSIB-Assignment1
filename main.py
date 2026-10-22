from ucimlrepo import fetch_ucirepo
import pandas as pd

credit = fetch_ucirepo(id=144)

X = credit.data.features
y = credit.data.targets

print("Dataset shape:", X.shape)
print("\nColumns:\n", X.columns.tolist())
print("\nFirst five rows:\n", X.head())
print("\nData types:\n", X.dtypes)
print("\nMissing values:\n", X.isnull().sum())
print("\nTarget distribution:\n", y.value_counts())

df = pd.concat([X, y], axis=1)
df.to_csv("data/german_credit.csv", index=False)

print("\nDataset saved to data/german_credit.csv")