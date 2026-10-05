import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor


RANDOM_STATE = 42


# ============================================================
# WCZYTANIE DANYCH
# ============================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


print(train.shape)
print(test.shape)



# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(df):

    df = df.copy()


    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0)
        +
        df["1stFlrSF"].fillna(0)
        +
        df["2ndFlrSF"].fillna(0)
    )


    df["TotalBathrooms"] = (
        df["FullBath"].fillna(0)
        +
        0.5 * df["HalfBath"].fillna(0)
        +
        df["BsmtFullBath"].fillna(0)
        +
        0.5 * df["BsmtHalfBath"].fillna(0)
    )


    df["OverallScore"] = (
        df["OverallQual"]
        *
        df["GrLivArea"]
    )


    df["Age"] = (
        df["YrSold"]
        -
        df["YearBuilt"]
    )


    df["RemodAge"] = (
        df["YrSold"]
        -
        df["YearRemodAdd"]
    )


    df["GarageScore"] = (
        df["GarageCars"].fillna(0)
        *
        df["GarageArea"].fillna(0)
    )


    df["TotalPorch"] = (
        df["WoodDeckSF"].fillna(0)
        +
        df["OpenPorchSF"].fillna(0)
        +
        df["EnclosedPorch"].fillna(0)
        +
        df["3SsnPorch"].fillna(0)
        +
        df["ScreenPorch"].fillna(0)
    )


    df["TotalArea"] = (
        df["TotalBsmtSF"].fillna(0)
        +
        df["1stFlrSF"].fillna(0)
        +
        df["2ndFlrSF"].fillna(0)
        +
        df["GarageArea"].fillna(0)
    )


    df["HasGarage"] = (
        df["GarageArea"].fillna(0) > 0
    ).astype(int)


    df["HasPool"] = (
        df["PoolArea"].fillna(0) > 0
    ).astype(int)


    return df



X = train.drop(
    columns=["SalePrice"]
)

y = train["SalePrice"]


X_test = test.copy()



X = feature_engineering(X)

X_test = feature_engineering(X_test)


# ============================================================
# UZUPEŁNIANIE BRAKÓW DLA CATBOOST
# ============================================================

# kolumny tekstowe
cat_columns = X.select_dtypes(
    exclude=np.number
).columns


for col in cat_columns:

    X[col] = X[col].fillna("Missing").astype(str)

    X_test[col] = X_test[col].fillna("Missing").astype(str)



# kolumny numeryczne
num_columns = X.select_dtypes(
    include=np.number
).columns


for col in num_columns:

    median = X[col].median()

    X[col] = X[col].fillna(median)

    X_test[col] = X_test[col].fillna(median)
# log target

y = np.log1p(y)



# ============================================================
# OUTLIERY
# ============================================================

outliers = X[
    (X["GrLivArea"] > 4000)
    &
    (y > 300000)
].index


print(
    "Liczba outlierów:",
    len(outliers)
)

print(
    outliers
)


X = X.drop(outliers)

y = y.drop(outliers)



# ============================================================
# LOG TARGET
# ============================================================

y = np.log1p(y)



# ============================================================
# KATEGORIE DLA CATBOOST
# ============================================================

cat_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()


print()
print("Kategorie:")
print(cat_features)

print(
    "Liczba kategorii:",
    len(cat_features)
)



# ============================================================
# MODEL CATBOOST
# ============================================================

cat = CatBoostRegressor(

    iterations=8000,

    learning_rate=0.01,

    depth=5,

    loss_function="RMSE",

    l2_leaf_reg=8,

    random_strength=0.5,

    bagging_temperature=0.2,

    random_seed=42,

    verbose=False

)



# ============================================================
# CROSS VALIDATION
# ============================================================

kf = KFold(

    n_splits=10,

    shuffle=True,

    random_state=42

)


scores = []


for train_idx, val_idx in kf.split(X):


    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]


    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]


    model = CatBoostRegressor(

        iterations=8000,

        learning_rate=0.01,

        depth=5,

        loss_function="RMSE",

        l2_leaf_reg=8,

        random_strength=0.5,

        bagging_temperature=0.2,

        random_seed=42,

        verbose=False

    )

    model.fit(

        X_train,

        y_train,

        cat_features=cat_features,

        eval_set=(X_val, y_val),

        early_stopping_rounds=300

    )


    pred = model.predict(
        X_val
    )


    r2 = model.score(
        X_val,
        y_val
    )


    scores.append(r2)


    print(
        f"Fold R2: {r2:.5f}"
    )



print()

print("="*60)

print(
    "ŚREDNIE R2:",
    np.mean(scores)
)

print(
    "STD:",
    np.std(scores)
)

print("="*60)



# ============================================================
# FINALNY TRENING
# ============================================================


final_model = CatBoostRegressor(

    iterations=8000,

    learning_rate=0.01,

    depth=5,

    loss_function="RMSE",

    l2_leaf_reg=8,

    random_strength=0.5,

    bagging_temperature=0.2,

    random_seed=42,

    verbose=False

)



final_model.fit(

    X,

    y,

    cat_features=cat_features

)



# ============================================================
# PREDYKCJA TEST
# ============================================================


pred_log = final_model.predict(
    X_test
)


pred = np.expm1(
    pred_log
)


pred = np.maximum(
    pred,
    0
)



submission = pd.DataFrame({

    "Id": test["Id"],

    "SalePrice": pred

})


submission.to_csv(

    "submission_catboost.csv",

    index=False

)


print()

print(submission.head())

print()

print("Zapisano submission_catboost.csv")