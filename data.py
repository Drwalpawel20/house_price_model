import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    RobustScaler
)

from sklearn.model_selection import (
    KFold,
    cross_val_score,
    RandomizedSearchCV
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (

    RandomForestRegressor,

    ExtraTreesRegressor,

    GradientBoostingRegressor,

    HistGradientBoostingRegressor,

    StackingRegressor

)

from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

import joblib

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.shape)
print(test.shape)

train.head()
y = train["SalePrice"]

X = train.drop(
    columns=["SalePrice"]
)
def feature_engineering(df):

    df = df.copy()

    df["TotalArea"] = (
            df["TotalBsmtSF"].fillna(0)
            +
            df["1stFlrSF"].fillna(0)
            +
            df["2ndFlrSF"].fillna(0)
            +
            df["GarageArea"].fillna(0)
    )

    df["TotalBathrooms"] = (
            df["FullBath"]
            +
            0.5 * df["HalfBath"]
            +
            df["BsmtFullBath"].fillna(0)
            +
            0.5 * df["BsmtHalfBath"].fillna(0)
    )

    df["TotalPorch"] = (
            df["OpenPorchSF"]
            +
            df["EnclosedPorch"]
            +
            df["3SsnPorch"]
            +
            df["ScreenPorch"]
            +
            df["WoodDeckSF"]
    )

    df["OverallQual_TotalSF"] = (
            df["OverallQual"]
            *
            df["TotalSF"]
    )

    df["GarageAge"] = (
            df["YrSold"]
            -
            df["GarageYrBlt"]
    )

    df["RemodAge"] = (
            df["YrSold"]
            -
            df["YearRemodAdd"]
    )

    df["HasGarage"] = (
            df["GarageArea"] > 0
    ).astype(int)

    df["HasPool"] = (
            df["PoolArea"] > 0
    ).astype(int)

    df["Has2ndFloor"] = (
            df["2ndFlrSF"] > 0
    ).astype(int)


    return df


X = feature_engineering(X)



X_test = test.copy()
X_test = feature_engineering(X_test)
y = np.log1p(y)

# usuwanie ekstremalnych outlierów

outliers = X[
    (X["GrLivArea"] > 4000)
    &
    (y > np.log1p(300000))
].index


print("Outliery:", outliers)


X = X.drop(outliers)
y = y.drop(outliers)

missing = pd.DataFrame({

    "Missing": X.isnull().sum(),

    "Percent":
        100 * X.isnull().mean()

})

missing = missing[
    missing["Missing"] > 0
]

missing = missing.sort_values(
    "Percent",
    ascending=False
)

print(missing)

plt.figure(figsize=(10,8))

sns.barplot(

    data=missing,

    y=missing.index,

    x="Percent"

)

plt.title("Braki danych")

plt.tight_layout()

plt.show()
numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

print()

print("Numeryczne:", len(numeric_features))

print("Kategorialne:", len(categorical_features))
numeric_transformer = Pipeline(

    steps=[

        (

            "imputer",

            SimpleImputer(
                strategy="median"
            )

        ),

        (

            "scaler",

            RobustScaler()

        )

    ]

)
categorical_transformer = Pipeline(

    steps=[

        (

            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )

        ),

        (

            "encoder",

            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )

        )

    ]

)
preprocessor = ColumnTransformer(

    transformers=[

        (

            "num",

            numeric_transformer,

            numeric_features

        ),

        (

            "cat",

            categorical_transformer,

            categorical_features

        )

    ]

)
X_train = preprocessor.fit_transform(X)

X_test_final = preprocessor.transform(X_test)

print(X_train.shape)

print(X_test_final.shape)
kf = KFold(

    n_splits=10,

    shuffle=True,

    random_state=42

)
from sklearn.base import clone
# ============================================================
# AUTOMATYCZNE USUWANIE NAJGORSZYCH CECH EXTRA TREES
# ============================================================

from sklearn.base import clone

print("\n")
print("="*70)
print("ITERACYJNA SELEKCJA CECH - ExtraTrees")
print("="*70)


# model bazowy do oceny
feature_selector_model = ExtraTreesRegressor(

    n_estimators=800,
    max_depth=30,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1

)


# aktualny zestaw cech
X_selected = X_train.copy()

# testowy zestaw
X_test_selected = X_test_final.copy()


# nazwy po one-hot encoding
feature_names = (
    preprocessor
    .get_feature_names_out()
)


selected_features = list(feature_names)


# wynik początkowy
feature_selector_model.fit(
    X_selected,
    y
)


scores = cross_val_score(

    feature_selector_model,

    X_selected,

    y,

    cv=kf,

    scoring="r2",

    n_jobs=-1

)


best_r2 = scores.mean()


print(
    f"Początkowe cechy: {X_selected.shape[1]}"
)

print(
    f"Początkowe R2: {best_r2:.5f}"
)



iteration = 1



while True:


    print()
    print("-"*70)
    print(f"ITERACJA {iteration}")


    # --------------------------------------------------------
    # 1. Trenuj ExtraTrees
    # --------------------------------------------------------

    feature_selector_model.fit(
        X_selected,
        y
    )


    importances = pd.Series(

        feature_selector_model.feature_importances_,

        index=selected_features

    )


    # --------------------------------------------------------
    # 2. Znajdź 5 najmniej ważnych
    # --------------------------------------------------------

    remove_features = (

        importances

        .sort_values()

        .head(1)

        .index

        .tolist()

    )


    print(
        "Usuwam:"
    )

    for f in remove_features:
        print(" -", f)



    # indeksy zostających kolumn

    keep_indexes = [

        i

        for i,f

        in enumerate(selected_features)

        if f not in remove_features

    ]


    X_new = X_selected[:, keep_indexes]


    X_test_new = X_test_selected[:, keep_indexes]


    new_feature_names = [

        f

        for f in selected_features

        if f not in remove_features

    ]



    # --------------------------------------------------------
    # 3. Liczymy CV
    # --------------------------------------------------------

    cv_model = ExtraTreesRegressor(

        n_estimators=800,

        max_depth=30,

        min_samples_leaf=2,

        max_features="sqrt",

        random_state=42,

        n_jobs=-1

    )


    scores = cross_val_score(

        cv_model,

        X_new,

        y,

        cv=kf,

        scoring="r2",

        n_jobs=-1

    )


    new_r2 = scores.mean()


    print(
        f"Stare R2: {best_r2:.5f}"
    )

    print(
        f"Nowe R2:  {new_r2:.5f}"
    )



    # --------------------------------------------------------
    # 4. Decyzja
    # --------------------------------------------------------

    if new_r2 > best_r2:


        print(
            "POPRAWA -> zostawiam usunięcie"
        )


        X_selected = X_new

        X_test_selected = X_test_new

        selected_features = new_feature_names


        best_r2 = new_r2



    else:


        print(
            "BRAK POPRAWY -> koniec"
        )

        break



    iteration += 1



print()
print("="*70)

print("KONIEC SELEKCJI")

print("="*70)


print(
    "Najlepsze R2:",
    best_r2
)


print(
    "Pozostało cech:",
    X_selected.shape[1]
)



print("\nUsunięte cechy:")

removed = (

    set(feature_names)

    -

    set(selected_features)

)


for f in removed:
    print("-",f)



# zapisujemy finalne dane

X_train_final = X_selected

X_test_final = X_test_selected

ridge = Ridge(

    alpha=25,

    random_state=42

)
elastic = ElasticNet(

    alpha=0.001,

    l1_ratio=0.4,

    max_iter=10000,

    random_state=42

)
lasso = Lasso(

    alpha=0.0005,

    max_iter=10000,

    random_state=42

)
forest = RandomForestRegressor(

    n_estimators=1200,

    max_depth=20,

    min_samples_leaf=2,

    min_samples_split=5,

    max_features="sqrt",

    bootstrap=True,

    random_state=42,

    n_jobs=-1

)
extra = ExtraTreesRegressor(

    n_estimators=1500,

    max_depth=30,

    min_samples_leaf=2,

    max_features="sqrt",

    random_state=42,

    n_jobs=-1

)
gbr = GradientBoostingRegressor(

    learning_rate=0.015,

    n_estimators=3000,

    max_depth=2,

    min_samples_leaf=5,

    min_samples_split=10,

    max_features="sqrt",

    subsample=0.8,

    random_state=42

)
hist = HistGradientBoostingRegressor(

    learning_rate=0.03,

    max_depth=6,

    max_iter=700,

    l2_regularization=2,

    random_state=42

)
svr = SVR(

    kernel="rbf",

    C=30,

    epsilon=0.05,

    gamma="scale"

)
mlp = MLPRegressor(

    hidden_layer_sizes=(256,128,64),

    alpha=0.001,

    learning_rate_init=0.0005,

    max_iter=3000,

    early_stopping=True,

    random_state=42

)
xgb = XGBRegressor(

    learning_rate=0.01,

    n_estimators=8000,

    max_depth=2,

    min_child_weight=5,

    subsample=0.7,

    colsample_bytree=0.7,

    reg_alpha=1,

    reg_lambda=10,

    random_state=42

)
lgb = LGBMRegressor(

    learning_rate=0.01,

    n_estimators=5000,

    num_leaves=31,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.2,

    reg_lambda=2,

    random_state=42

)
cat = CatBoostRegressor(

    iterations=8000,
    learning_rate=0.01,
    depth=5,
    loss_function="RMSE",
    l2_leaf_reg=8,
    random_seed=42,
    verbose=False

)



models = {

    "Ridge": ridge,

    "ElasticNet": elastic,

    "Lasso": lasso,

    "RandomForest": forest,

    "ExtraTrees": extra,

    "GradientBoosting": gbr,

    "HistGradientBoosting": hist,

    "SVR": svr,

    "MLP": mlp,

    "XGBoost": xgb,

    "CatBoost": cat

}
results = {}

best_model = None

best_name = None

best_score = -999
for name, model in models.items():

    print()

    print("="*60)

    print(name)

    scores = cross_val_score(
        model,
        X_train_final,
        y,
        cv=kf,
        scoring="r2",
        n_jobs=-1
    )

    mean_score = scores.mean()

    std_score = scores.std()

    results[name] = mean_score

    print("Fold scores:")

    print(scores)

    print()

    print(

        f"Mean R2 = {mean_score:.5f}"

    )

    print(

        f"Std = {std_score:.5f}"

    )

    if mean_score > best_score:

        best_score = mean_score

        best_name = name

        best_model = clone(model)

print()

print("="*60)

print("PODSUMOWANIE")

print("="*60)

ranking = pd.DataFrame({

    "Model": results.keys(),

    "R2": results.values()

})

ranking = ranking.sort_values(

    "R2",

    ascending=False

)

print(ranking)

# ============================================================
# FINALNY TRENING NAJLEPSZEGO MODELU I PREDYKCJA TEST
# ============================================================

print()
print("="*70)
print("FINALNY MODEL:", best_name)
print("="*70)


# trening najlepszego modelu na całym zbiorze treningowym

best_model.fit(
    X_train_final,
    y
)


# predykcja (ciągle w log skali)

pred_log = best_model.predict(
    X_test_final
)


# cofnięcie log1p

pred_price = np.expm1(
    pred_log
)


# zabezpieczenie przed ujemnymi cenami

pred_price = np.maximum(
    pred_price,
    0
)



# ============================================================
# GENEROWANIE SUBMISSION
# ============================================================


submission = pd.DataFrame({

    "Id": test["Id"],

    "SalePrice": pred_price

})


submission.to_csv(

    "submission.csv",

    index=False

)


print()

print("Gotowe!")
print(
    submission.head()
)

print()

print(
    "Zapisano: submission.csv"
)