import pandas as pd 

df = pd.read_csv("train.csv")

print(df.head())

print("\Dataset Shape: ", df.shape)

print("\nColumns: ", df.columns.to_list())

print("\nMissing Values: ")
print(df.isnull().sum().sort_values(ascending=False).head(20))

df =df.drop(columns=[
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "MasVnrType",
    "FireplaceQu"
])

print(df.shape)
print(df.isnull().sum().sort_values(ascending=False).head(15))

df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["GarageYrBlt"].median())
df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].median())

garage_cols = ["GarageQual", "GarageType", "GarageCond", "GarageFinish"]
bsmt_cols = ["BsmtFinType1", "BsmtFinType2", "BsmtQual", "BsmtCond", "BsmtExposure"]


for col in garage_cols:
    df[col] = df[col].fillna("None")

for col in bsmt_cols:
    df[col] = df[col].fillna("None")

df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

print(df.isnull().sum().sum())
print(df.isnull().sum().sort_values(ascending=False).head(10))

print(df["SalePrice"].describe())

correlation = df.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)

print(correlation.head(15))