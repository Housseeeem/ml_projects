"""
model_pipeline.py
=================
Pipeline ML modulaire issu du notebook « Data_Preparation_3IA_1_learners ».

Objectif : prédire la durée de séjour en soins intensifs (``los_seconds``)
à partir des tables MIMIC INPUTEVENTS_CV1, ICUSTAYS et D_ITEMS.

Fonctions publiques (demandées par l'énoncé) :
    - prepare_data()   : charger + prétraiter les données
    - train_model()    : entraîner le modèle
    - evaluate_model() : évaluer les performances
    - save_model()     : sauvegarder le modèle entraîné
    - load_model()     : charger un modèle sauvegardé
"""

from __future__ import annotations

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Constantes (reprises du notebook)
# --------------------------------------------------------------------------- #
TARGET = "los_seconds"
TARGET_DATE = pd.Timestamp("2024-01-01")  # date de recalage des timestamps

INPUTEVENTS_DROP = [
    "row_id", "cgid", "orderid", "linkorderid", "originalsite", "0",
    "newbottle", "stopped", "originalrate", "originalrateuom", "rate",
    "rateuom", "originalroute",
]
ICUSTAYS_DROP = [
    "dbsource", "first_careunit", "last_careunit",
    "first_wardid", "last_wardid", "row_id",
]
ITEMS_DROP = [
    "row_id", "abbreviation", "dbsource", "linksto",
    "param_type", "conceptid", "category", "unitname",
]
# Colonnes d'identifiants / dates supprimées avant l'apprentissage
FINAL_DROP = [
    "subject_id", "hadm_id", "icustay_id", "charttime", "itemid",
    "storetime", "intime", "outtime", "amountuom", "label", "los", "time_diff",
]


# --------------------------------------------------------------------------- #
# Étapes de nettoyage (fonctions internes)
# --------------------------------------------------------------------------- #
def _shift_dates(s: pd.Series, offset: pd.Timedelta) -> pd.Series:
    """Convertit en datetime puis décale de ``offset``."""
    return pd.to_datetime(s, errors="coerce") - offset


def _fill_with_fallback_then_mode(df: pd.DataFrame, col: str, fallback: str) -> pd.DataFrame:
    """Remplit ``col`` par ``fallback``, puis par la valeur la plus fréquente."""
    df[col] = df[col].fillna(df[fallback])
    if df[col].notna().any():
        df[col] = df[col].fillna(df[col].mode()[0])
    return df.dropna(subset=[col])


def _clean_inputevents(inputevents: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timedelta]:
    df = inputevents.drop_duplicates().drop(columns=INPUTEVENTS_DROP, errors="ignore").copy()

    # Recalage de charttime : le max devient TARGET_DATE
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    offset = df["charttime"].max() - TARGET_DATE
    df["charttime"] = df["charttime"] - offset
    df = df.dropna(subset=["charttime"])

    # storetime : même décalage (comme dans le notebook)
    df["storetime"] = _shift_dates(df["storetime"], offset)
    df = df.dropna(subset=["storetime"])

    # Valeurs manquantes de amountuom / amount
    df = _fill_with_fallback_then_mode(df, "amountuom", "originalamountuom")
    df = _fill_with_fallback_then_mode(df, "amount", "originalamount")
    df = df.drop(columns=["originalamount", "originalamountuom"], errors="ignore")

    # Identifiants en entiers nullable, puis suppression des lignes sans id
    ids = ["itemid", "icustay_id", "subject_id", "hadm_id"]
    for c in ids:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df.dropna(subset=ids), offset


def _clean_icustays(icustays: pd.DataFrame, offset: pd.Timedelta) -> pd.DataFrame:
    df = icustays.drop_duplicates().drop(columns=ICUSTAYS_DROP, errors="ignore").copy()
    for c in ("intime", "outtime"):
        df[c] = _shift_dates(df[c], offset)
        df = df.dropna(subset=[c])
    df["los"] = df["los"].fillna((df["outtime"] - df["intime"]).dt.days)
    return df.dropna(subset=["los"])


def _clean_items(items: pd.DataFrame) -> pd.DataFrame:
    df = items.drop_duplicates().drop(columns=ITEMS_DROP, errors="ignore")
    return df.dropna(subset=["label"])


def _build_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Encodage, features temporelles et construction de la cible."""
    df = merged.copy()

    # Encodage des variables catégorielles
    df["label_encoded"] = LabelEncoder().fit_transform(df["label"])
    df["amountuom_encoded"] = LabelEncoder().fit_transform(df["amountuom"])

    # Numéro du traitement par patient
    df = df.sort_values(by=["subject_id", "charttime"])
    df["treatment_number"] = df.groupby("subject_id").cumcount() + 1

    # time_diff : 1er traitement -> charttime - intime ; sinon écart avec le précédent
    previous = df.groupby("subject_id")["charttime"].diff()
    df["time_diff"] = previous.where(df["treatment_number"] > 1, df["charttime"] - df["intime"])
    df["time_diff_seconds"] = df["time_diff"].dt.total_seconds()

    # Cible : durée de séjour en secondes (los est en jours)
    df[TARGET] = (pd.to_numeric(df["los"], errors="coerce") * 86400).round()

    df = df.drop(columns=FINAL_DROP, errors="ignore")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna().astype(float)
    return df


# --------------------------------------------------------------------------- #
# Fonctions publiques
# --------------------------------------------------------------------------- #
def prepare_data(
    inputevents_path: str = "INPUTEVENTS_CV1.csv",
    icustays_path: str = "ICUSTAYS.csv",
    items_path: str = "D_ITEMS.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
):
    """Charge, nettoie, fusionne et prépare les données.

    Retourne ``X_train, X_test, y_train, y_test, scaler`` (``scaler`` vaut
    ``None`` si ``scale=False``). Le scaler est ajusté sur le train uniquement.
    """
    inputevents = pd.read_csv(inputevents_path)
    icustays = pd.read_csv(icustays_path)
    items = pd.read_csv(items_path)

    inputevents2, offset = _clean_inputevents(inputevents)
    icustays2 = _clean_icustays(icustays, offset)
    items2 = _clean_items(items)

    merged = inputevents2.merge(icustays2, on=["subject_id", "hadm_id", "icustay_id"], how="inner")
    merged = merged.merge(items2, on="itemid", how="inner")

    data = _build_features(merged)
    X, y = data.drop(columns=[TARGET]), data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = None
    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train, y_train, model_type: str = "linear", random_state: int = 42):
    """Entraîne un modèle de régression (``linear`` ou ``random_forest``)."""
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"model_type inconnu : {model_type!r}")
    return model.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test) -> dict:
    """Retourne MAE, RMSE et R² sur le jeu de test."""
    y_pred = model.predict(X_test)
    return {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
    }


def save_model(model, path: str = "model.joblib", scaler=None) -> str:
    """Sauvegarde le modèle (et le scaler s'il est fourni) dans un fichier joblib."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)
    return path


def load_model(path: str = "model.joblib"):
    """Charge un modèle sauvegardé. Retourne ``(model, scaler)``."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Aucun modèle trouvé : {path}")
    bundle = joblib.load(path)
    return bundle["model"], bundle["scaler"]
