#!/usr/bin/env python3
import os
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def download_dataset(url: str, local_path: str) -> str:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        print(f"Descargando dataset desde {url} a {local_path} ...")
        df = pd.read_csv(url, sep=';')
        df.to_csv(local_path, index=False)
    else:
        print(f"Dataset ya existe en {local_path}")
    return local_path

def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)
    # limpieza básica
    df = df.dropna()
    # convertir a problema de clasificación binaria: quality >= 6 => 1
    df['target'] = (df['quality'] >= 6).astype(int)
    X = df.drop(columns=['quality', 'target'])
    y = df['target']
    return X, y

def run_train(config: dict, verbose: bool = True) -> dict:
    # Paths y configuración
    dataset_cfg = config.get('dataset', {})
    model_cfg = config.get('model', {})
    mlflow_cfg = config.get('mlflow', {})
    output_cfg = config.get('output', {})
    train_cfg = config.get('training', {})

    local_csv = download_dataset(dataset_cfg['url'], dataset_cfg.get('local_path', 'data/winequality-red.csv'))
    X, y = load_and_preprocess(local_csv)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_cfg.get('test_size', 0.2), random_state=train_cfg.get('random_state', 42), stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=int(model_cfg.get('n_estimators', 100)),
        max_depth=int(model_cfg.get('max_depth', 7)),
        random_state=int(model_cfg.get('random_state', 42)),
        n_jobs=-1
    )

    # MLflow tracking local
    tracking_dir = os.path.abspath(mlflow_cfg.get('tracking_dir', 'mlruns'))
    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    exp_name = mlflow_cfg.get('experiment_name', 'default')
    mlflow.set_experiment(exp_name)

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            'n_estimators': model_cfg.get('n_estimators'),
            'max_depth': model_cfg.get('max_depth'),
            'test_size': train_cfg.get('test_size'),
            'random_state': train_cfg.get('random_state'),
        })

        # Entrenar
        clf.fit(X_train_scaled, y_train)

        # Predicción y métricas
        preds = clf.predict(X_test_scaled)
        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, zero_division=0))

        mlflow.log_metrics({'accuracy': acc, 'f1': f1})

        # Firma y ejemplo de entrada
        input_example = X_train.iloc[:3]
        try:
            signature = infer_signature(input_example, clf.predict_proba(scaler.transform(input_example)) )
        except Exception:
            # fallback: use predictions
            signature = infer_signature(input_example, clf.predict(scaler.transform(input_example)))

        # Guardar modelo (mlflow + joblib)
        model_dir = output_cfg.get('model_dir', 'models')
        model_name = output_cfg.get('model_name', 'model')
        os.makedirs(model_dir, exist_ok=True)

        # Guardar el pipeline simple: scaler + clf
        pipeline = {'scaler': scaler, 'model': clf}
        joblib.dump(pipeline, os.path.join(model_dir, f"{model_name}.pkl"))

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path=model_name,
            registered_model_name=None,
            signature=signature,
            input_example=input_example
        )

        if verbose:
            print(f"Run ID: {run.info.run_id}")
            print(f"Accuracy: {acc:.4f}  F1: {f1:.4f}")
            print(f"Modelo guardado en {os.path.join(model_dir, model_name + '.pkl')}")
            print(f"MLflow tracking dir: {tracking_dir}")

        return {'accuracy': acc, 'f1': f1, 'run_id': run.info.run_id}

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento pipeline ML con MLflow")
    parser.add_argument("--config", "-c", default="config.yaml", help="Ruta al archivo de configuración YAML")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_train(cfg)