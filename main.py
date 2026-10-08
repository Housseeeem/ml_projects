"""main.py — exécution du pipeline : préparation -> entraînement -> évaluation -> sauvegarde.

Exemples :
    python main.py                       # pipeline complet
    python main.py --model random_forest
    python main.py --load                # recharge model.joblib et l'évalue seulement
"""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline ML - prédiction de la durée de séjour (los)")
    p.add_argument("--inputevents", default="INPUTEVENTS_CV1.csv")
    p.add_argument("--icustays", default="ICUSTAYS.csv")
    p.add_argument("--items", default="D_ITEMS.csv")
    p.add_argument("--model", default="linear", choices=["linear", "random_forest"])
    p.add_argument("--model-path", default="model.joblib")
    p.add_argument("--load", action="store_true", help="charger le modèle existant au lieu d'entraîner")
    return p.parse_args()


def main():
    args = parse_args()

    print("[1/4] Préparation des données...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        args.inputevents, args.icustays, args.items
    )
    print(f"      train={X_train.shape}  test={X_test.shape}")

    if args.load:
        print(f"[2/4] Chargement du modèle : {args.model_path}")
        model, _ = load_model(args.model_path)
    else:
        print(f"[2/4] Entraînement ({args.model})...")
        model = train_model(X_train, y_train, model_type=args.model)

    print("[3/4] Évaluation...")
    for name, value in evaluate_model(model, X_test, y_test).items():
        print(f"      {name}: {value:.4f}")

    if not args.load:
        print(f"[4/4] Sauvegarde -> {save_model(model, args.model_path, scaler)}")


if __name__ == "__main__":
    main()
