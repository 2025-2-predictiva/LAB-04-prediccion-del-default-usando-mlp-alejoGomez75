import os
import json
import gzip
import pickle
from glob import glob
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ===================== CARGA Y PREPARACIÓN DE DATOS ===================== #

def cargar_y_preparar(ruta_zip: str) -> pd.DataFrame:
    """
    Lee el .csv.zip, ajusta columnas clave y aplica filtros básicos.
    """
    df = pd.read_csv(ruta_zip, compression="zip").copy()

    # Homogeneizar nombre de la variable respuesta
    df.rename(columns={"default payment next month": "default"}, inplace=True)

    # ID no aporta información para el modelo
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)

    # Se descartan filas con categorías 0 en estas variables
    df = df[(df["MARRIAGE"] != 0) & (df["EDUCATION"] != 0)].copy()

    # Agrupar niveles altos de EDUCATION en un solo valor
    df["EDUCATION"] = df["EDUCATION"].apply(lambda v: 4 if v >= 4 else v)

    # Se eliminan registros incompletos
    return df.dropna()


# ===================== MÉTRICAS Y MATRICES ===================== #

def resumen_metricas(etiqueta: str, y_true, y_pred) -> Dict:
    """
    Construye un diccionario con las métricas más usadas.
    """
    return {
        "type": "metrics",
        "dataset": etiqueta,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def matriz_confusion_dict(etiqueta: str, y_true, y_pred) -> Dict:
    """
    Convierte la matriz de confusión en un formato más amigable para guardar.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "type": "cm_matrix",
        "dataset": etiqueta,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
    }


# ===================== PIPELINE Y GRIDSEARCH ===================== #

def crear_busqueda(vars_cat, vars_num) -> GridSearchCV:
    """
    Arma el pipeline completo (prepro + selección + PCA + MLP)
    y define el espacio de búsqueda para GridSearchCV.
    """
    transformador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), vars_cat),
            ("num", StandardScaler(), vars_num),
        ]
    )

    pipeline_mlp = Pipeline(
        steps=[
            ("pre", transformador),
            ("selector", SelectKBest(score_func=f_classif)),
            ("pca", PCA()),
            ("mlp", MLPClassifier(max_iter=15000, random_state=21)),
        ]
    )

    # Configuración de hiperparámetros ya acotada a una región “buena”
    grid = {
        "selector__k": [20],
        "pca__n_components": [None],
        "mlp__hidden_layer_sizes": [(50, 30, 40, 60)],
        "mlp__alpha": [0.26],
        "mlp__learning_rate_init": [0.001],
    }

    return GridSearchCV(
        estimator=pipeline_mlp,
        param_grid=grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )


# ===================== PUNTO DE ENTRADA ===================== #

def main() -> None:
    ruta_train = "files/input/train_data.csv.zip"
    ruta_test = "files/input/test_data.csv.zip"

    df_tr = cargar_y_preparar(ruta_train)
    df_te = cargar_y_preparar(ruta_test)

    X_tr, y_tr = df_tr.drop(columns=["default"]), df_tr["default"]
    X_te, y_te = df_te.drop(columns=["default"]), df_te["default"]

    columnas_cat = ["SEX", "EDUCATION", "MARRIAGE"]
    columnas_num = [c for c in X_tr.columns if c not in columnas_cat]

    buscador = crear_busqueda(columnas_cat, columnas_num)
    buscador.fit(X_tr, y_tr)

    modelos_dir = Path("files/models")
    if modelos_dir.exists():
        # Limpio cualquier modelo previo para dejar solo el actual
        for ruta in glob(str(modelos_dir / "*")):
            os.remove(ruta)
        try:
            os.rmdir(modelos_dir)
        except OSError:
            # Si la carpeta no se puede borrar (por algo residual), se ignora
            pass
    modelos_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(modelos_dir / "model.pkl.gz", "wb") as fh:
        pickle.dump(buscador, fh)

    y_tr_pred = buscador.predict(X_tr)
    y_te_pred = buscador.predict(X_te)

    m_train = resumen_metricas("train", y_tr, y_tr_pred)
    m_test = resumen_metricas("test", y_te, y_te_pred)
    cm_train = matriz_confusion_dict("train", y_tr, y_tr_pred)
    cm_test = matriz_confusion_dict("test", y_te, y_te_pred)

    resultados: List[Dict] = [m_train, m_test, cm_train, cm_test]

    out_dir = Path("files/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        for registro in resultados:
            f.write(json.dumps(registro) + "\n")


if __name__ == "__main__":
    main()