"""
Files:
- Input CSV: WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

import os
import io
import json
import math
import time
import requests
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

LOCAL_CSV = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# AWS Config
S3_BUCKET = os.environ.get("MODEL_BUCKET", "simon-hong-telco-model")
S3_KEY_LIGHT = os.environ.get("MODEL_KEY", "models/telco_churn_light.json")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

# Flag: make AWS completely optional
#   - False: run everything locally; no S3 upload
#   - True : in addition, build a lightweight artifact and upload to S3
ENABLE_AWS_EXPORT = True

if ENABLE_AWS_EXPORT:
    import boto3


# Data Telco CSV and Preprocessing
# - convert TotalCharges to numeric
# - convert Churn to 0/1
# - drop customerID
def load_telco(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=["customerID"], errors="ignore")
    return df

# Preprocessor
# - numeric: impute median, standard scale
# - categorical: impute most frequent, one-hot encode
def build_preprocessor(df: pd.DataFrame):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, X, y, numeric_features, categorical_features

# Prepare Numpy for Keras
# - fit preprocessor on X_train
# - transform X_train and X_val
# - return dense numpy arrays for Keras
def make_numpy_for_keras(preprocessor, X_train, X_val):
    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out()
    num_names = preprocessor.transformers_[0][2]
    feature_names = np.concatenate([num_names, cat_names])

    return X_train_proc.astype("float32"), X_val_proc.astype("float32"), feature_names

# Keras Gradient Tracker
class GradientTracker(keras.callbacks.Callback):
    def __init__(self, model, sample_x, sample_y):
        super().__init__()
        self.model_for_grads = model
        self.sample_x = tf.convert_to_tensor(sample_x)
        self.sample_y = tf.convert_to_tensor(sample_y, dtype=tf.float32)
        self.history = []

# Track gradients at the end of each epoch
    def on_epoch_end(self, epoch, logs=None):
        with tf.GradientTape() as tape:
            preds = self.model_for_grads(self.sample_x, training=True)  # (batch, 1)
            preds = tf.squeeze(preds, axis=-1)  # -> (batch,)
            loss = keras.losses.binary_crossentropy(self.sample_y, preds)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.model_for_grads.trainable_weights)
        grad_vals = [tf.reshape(g, [-1]) for g in grads if g is not None]
        if grad_vals:
            all_grads = tf.concat(grad_vals, axis=0)
            grad_mean = tf.reduce_mean(tf.abs(all_grads)).numpy().item()
            grad_max = tf.reduce_max(tf.abs(all_grads)).numpy().item()
            grad_std = tf.math.reduce_std(all_grads).numpy().item()
        else:
            grad_mean = grad_max = grad_std = 0.0

        self.history.append(
            {
                "epoch": int(epoch),
                "grad_abs_mean": grad_mean,
                "grad_abs_max": grad_max,
                "grad_std": grad_std,
                "loss": float(loss.numpy()),
                "logs": logs or {},
            }
        )


# ##################
# Keras Model
def build_keras_model(input_dim: int):
    """""
    Return:
      - a compiled Keras model ready for training.
    """

    model = keras.Sequential()
    # specify input layer
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(16,activation="relu"))
    model.add(layers.Dense(8, activation="relu"))
    model.add(layers.Dense(1,activation="sigmoid"))
    #compile
    adam_optimizer= keras.optimizers.Adam(learning_rate=0.003)
    model.compile(loss="binary_crossentropy",optimizer=adam_optimizer,metrics=["AUC","accuracy"])
    return model

############################
# Scikit-Learn Models
def train_sklearn_models(preprocessor, X, y):
    # 1. Split X, y into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # 2. Build and train models
    # LogisticRegression
    LR = Pipeline(steps = [("preprocessor", preprocessor), ("clf",LogisticRegression(max_iter=500))])
    LR.fit(X_train,y_train)
    LR_proba = LR.predict_proba(X_val)[:,1]
    LR_AUC = roc_auc_score(y_val, LR_proba)
    results =[]
    results.append(("LogisticRegression",LR,LR_AUC))
    # RandomForestClassifier
    rfc = Pipeline(steps=[("preprocessor",preprocessor),("clf",RandomForestClassifier(n_estimators=100, random_state=42))])
    rfc.fit(X_train, y_train)
    rfc_proba = rfc.predict_proba(X_val)[:,1]
    rfc_AUC = roc_auc_score(y_val, rfc_proba)
    results.append(("RandomForestClassifier", rfc, rfc_AUC))
    # GradientBoostingClassifier
    gbc = Pipeline(steps=[("preprocessor", preprocessor),("clf",GradientBoostingClassifier(n_estimators=100, random_state=42))])
    gbc.fit(X_train,y_train)
    gbc_proba = gbc.predict_proba(X_val)[:,1]
    gbc_AUC = roc_auc_score(y_val, gbc_proba)
    results.append(("GradientBoostingClassifier", gbc, gbc_AUC))
    # SGDClassifier
    sgdc = Pipeline(steps=[("preprocessor",preprocessor),("clf", SGDClassifier(loss="log_loss", random_state=42))])
    sgdc.fit(X_train, y_train)
    sgdc_proba = sgdc.predict_proba(X_val)[:,1]
    sgdc_AUC = roc_auc_score(y_val, sgdc_proba)
    results.append(("SGDClassifier", sgdc, sgdc_AUC))

    # result list
    results = sorted(results, key=lambda x: x[2], reverse=True)
    return results, (X_train, X_val, y_train, y_val), sgdc, sgdc_AUC


################################################
# 4) Lightweight Artifact for AWS
# Artifact - plain python dict
# - contain only essential information to reconstruct preprocessing + SGD classifier
def build_lightweight_artifact(
    df,
    preprocessor,
    numeric_features,
    categorical_features,
    sgd_model,
):
    # 1. Fit preprocessor on full data
    X_full = df.drop("Churn", axis=1)
    preprocessor.fit(X_full)
    # 2. Extract numeric imputer and scaler parameters
    num_pipe = preprocessor.named_transformers_["num"]
    imputer  = num_pipe.named_steps["imputer"]
    scaler   = num_pipe.named_steps["scaler"]

    numeric_medians = imputer.statistics_.tolist()
    numeric_means = scaler.mean_.tolist()
    numeric_scales = scaler.scale_.tolist()

    # 3. Extract categorical one-hot encoder info
    cat_pipe = preprocessor.named_transformers_["cat"]
    ohe = cat_pipe.named_steps["onehot"]
    categories = [c.tolist() for c in ohe.categories_]

    # 4. Extract SGD classifier coefficients and intercept
    clf = sgd_model.named_steps["clf"]
    coef = clf.coef_.tolist()
    intercept = clf.intercept_.tolist()

    # 5. Construct the artifact dict
    artifact = {
        "numeric_features": numeric_features,
        "numeric_imputer_medians": numeric_medians,
        "numeric_means": numeric_means,
        "numeric_scales": numeric_scales,
        "categorical_features": categorical_features,
        "categories": categories,
        "coef": coef,
        "intercept": intercept,
        }

    return artifact



#######################
# AWS Upload
def upload_json_to_s3(obj, bucket, key, region="us-east-2"):
    """
    Upload a JSON-serializable object to S3. Used only if ENABLE_AWS_EXPORT is True.
    """
    s3 = boto3.client("s3", region_name=region)
    body = json.dumps(obj).encode("utf-8")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    print(f"Uploaded to s3://{bucket}/{key}")

s3 = boto3.client("s3")
_ARTIFACT = None
_WEIGHTS = None
_INTERCEPT = None

# Load artifact from S3
def load_artifact():
    bucket = os.environ.get("MODEL_BUCKET")
    key = os.environ.get("MODEL_KEY")

    global _ARTIFACT, _WEIGHTS, _INTERCEPT

    if _ARTIFACT is not None:
        return _ARTIFACT
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")

    wrapped_artifact = json.loads(body)
    artifact = wrapped_artifact["artifact"]
    weights = artifact["coef"][0]
    intercept = artifact["intercept"][0]

    _ARTIFACT = artifact
    _WEIGHTS = weights
    _INTERCEPT = intercept

    return _ARTIFACT
# Parse input record using artifact
def parse_artifact(record, artifact):
    numeric_features = artifact["numeric_features"]
    numeric_medians = artifact["numeric_imputer_medians"]
    numeric_means = artifact["numeric_means"]
    numeric_scales = artifact["numeric_scales"]
    categorical_features = artifact["categorical_features"]
    categories = artifact["categories"]

    features = []
    # numeric features
    for i, name in enumerate(numeric_features):
        median = numeric_medians[i]
        mean = numeric_means[i]
        scale = numeric_scales[i]

        val = record.get(name, None)

        try:
            if val is None or val == "":
                val = float(median)
            else:
                val = float(val)
        except (ValueError, TypeError):
            val = float(median)

        if scale ==0:
            val_scaled = 0.0
        else:
            val_scaled = (val - mean) / scale
        features.append(val_scaled)

    # categorical features
    for j, feature_name in enumerate(categorical_features):
        categories = artifact["categories"][j]
        val = record.get(feature_name, None)

        if val is None:
            val = ""
        for cat in categories:
            features.append(1.0 if str(val) == str(cat) else 0.0)

    return features

# Predict probability using logistic regression formula
def lambda_predict_proba(features):
    global _WEIGHTS, _INTERCEPT
    if _WEIGHTS is None or _INTERCEPT is None:
        raise ValueError("Artifact not loaded properly.")
    b = _INTERCEPT
    for w,x in zip(_WEIGHTS, features):
        b += w * x
    prob = 1.0 / (1.0 + math.exp(-b))
    return prob
def lambda_handler(event, context):
    artifact = load_artifact()
    try:
        if "body" in event and isinstance(event["body"], str):
            record = json.loads(event["body"])
        else:
            record = event

        records = record.get("records", [])
        if not isinstance(records, list) or len(records) == 0:
            raise ValueError("Request must contain a non-empty 'records' list.")
    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f"Invalid input: {str(e)}"}),
        }
    predictions = []
    labels = []

    for r in records:
        features = parse_artifact(r, artifact)
        prob = lambda_predict_proba(features)
        label = int(prob>=0.5)

        predictions.append(prob)
        labels.append(label)

    response_body = { "predictions": predictions, "labels": labels }
    return {"statusCode": 200, "body": json.dumps(response_body)}




def main():
    # Step 0: Load data
    if not os.path.exists(LOCAL_CSV):
        raise FileNotFoundError(f"CSV not found at {LOCAL_CSV}")

    print("Loading telco data...")
    df = load_telco(LOCAL_CSV)

    # Step 1: Preprocessor
    print("Building preprocessor...")
    preprocessor, X, y, num_feats, cat_feats = build_preprocessor(df)

    # Step 2: Scikit-Learn models
    print("Training sklearn models...")
    sk_results, splits, sgd_model, sgd_auc = train_sklearn_models(preprocessor, X, y)
    (X_train, X_val, y_train, y_val) = splits

    # Get best sklearn model
    best_sklearn_name, best_sklearn_model, best_sklearn_auc = sk_results[0]

    # Step 3: Keras model
    print("Preparing numpy for Keras...")
    X_train_np, X_val_np, feature_names = make_numpy_for_keras(
        preprocessor, X_train, X_val
    )

    print("Building Keras model...")
    keras_model = build_keras_model(input_dim=X_train_np.shape[1])

    # Train Keras with Gradient Tracking
    sample_x = X_train_np[:256]
    sample_y = y_train.values[:256]
    grad_tracker = GradientTracker(keras_model, sample_x, sample_y)

    print("Training Keras model...")
    history = keras_model.fit(
        X_train_np,
        y_train.values,
        validation_data=(X_val_np, y_val.values),
        epochs=30,
        batch_size=256,
        callbacks=[grad_tracker],
        verbose=1,
    )

    # Save gradient stats
    with open("keras_gradients.json", "w") as f:
        json.dump(grad_tracker.history, f, indent=2)
    print("Saved gradient stats to keras_gradients.json")

    # Plot gradient over epochs
    epochs = [entry["epoch"] for entry in grad_tracker.history]
    grad_means = [entry["grad_abs_mean"] for entry in grad_tracker.history]
    grad_max = [entry["grad_abs_max"] for entry in grad_tracker.history]
    grad_std = [entry["grad_std"] for entry in grad_tracker.history]

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, grad_means, label= "Gradient Mean")
    plt.plot(epochs, grad_max, label="Gradient Max")
    plt.plot(epochs, grad_std, label="Gradient Std")
    plt.xlabel("Epochs")
    plt.ylabel("Gradient Magnitude")
    plt.title("Keras Model Gradient Statistics Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # Step 4: Evaluate Keras on validation set
    y_val_pred = keras_model.predict(X_val_np).ravel()
    keras_auc = roc_auc_score(y_val.values, y_val_pred)
    keras_acc = accuracy_score(y_val.values, (y_val_pred >= 0.5).astype(int))
    # Gradient quality metrics
    last_grad = grad_tracker.history[-1]
    grad_quality = {
        "last_epoch_grad_abs_mean": last_grad["grad_abs_mean"],
        "last_epoch_grad_abs_max": last_grad["grad_abs_max"],
        "last_epoch_grad_std": last_grad["grad_std"],
    }

    print("\n=== MODEL SUMMARY (local) ===")
    print(f"Best sklearn: {best_sklearn_name} AUC={best_sklearn_auc:.4f}")
    print(f"Keras model : AUC={keras_auc:.4f} ACC={keras_acc:.4f}")
    print(f"SGD (for potential Lambda): AUC={sgd_auc:.4f}")
    print(f"Gradient quality (keras): {grad_quality}")

    # Step 5: Choose a final model for reporting (policy is up to you)

    gradient_mean = grad_quality["last_epoch_grad_abs_mean"]
    gradient_max = grad_quality["last_epoch_grad_abs_max"]
    if (keras_auc >= best_sklearn_auc) and (gradient_mean<0.05 and gradient_mean>1e-5) and (gradient_max<1.0):
        final_model_name = "keras_model"
        final_auc = keras_auc
    else:
        final_model_name = best_sklearn_name
        final_auc = best_sklearn_auc

    print(f"\nFINAL CHOICE (for your report): {final_model_name} AUC={final_auc:.4f}")

    # Step 6: Build leaderboard
    model_leaderboard = []
    for name, model_obj, auc_val in sk_results:
        model_leaderboard.append(
            {
                "name": name,
                "type": "sklearn",
                "auc": float(auc_val),
            }
        )

    # Add Keras to leaderboard
    model_leaderboard.append(
        {
            "name": "keras_mlp",
            "type": "keras",
            "auc": float(keras_auc),
            "accuracy": float(keras_acc),
            "grad_abs_mean": float(grad_tracker.history[-1]["grad_abs_mean"]),
            "grad_abs_max": float(grad_tracker.history[-1]["grad_abs_max"]),
            "grad_std": float(grad_tracker.history[-1]["grad_std"]),
        }
    )

    # Save leaderboard to JSON
    with open("model_leaderboard_telco.json", "w") as f:
        json.dump(model_leaderboard, f, indent=2)
    print("Saved model_leaderboard_telco.json")

    # Step 7 : Build lightweight artifact for AWS Lambda
    if ENABLE_AWS_EXPORT:
        print("\nENABLE_AWS_EXPORT=True → building lightweight artifact and uploading to S3.")

        artifact = build_lightweight_artifact(
            df,
            preprocessor,
            num_feats,
            cat_feats,
            sgd_model,
        )

        wrapped_artifact = {
            "artifact": artifact,
            "training_meta": {
                "final_choice": final_model_name,
                "final_choice_auc": float(final_auc),
                "best_sklearn": best_sklearn_name,
                "best_sklearn_auc": float(best_sklearn_auc),
                "keras_auc": float(keras_auc),
                "keras_acc": float(keras_acc),
                "model_leaderboard": model_leaderboard,
            },
        }

        upload_json_to_s3(wrapped_artifact, S3_BUCKET, S3_KEY_LIGHT, AWS_REGION)
        print("Uploaded lightweight artifact with metadata to S3.")
    else:
        print(
            "\nENABLE_AWS_EXPORT=False → skipping S3 upload.\n"
            "You can enable AWS export later by:\n"
            "  1) Setting MODEL_BUCKET and MODEL_KEY environment variables.\n"
            "  2) Setting ENABLE_AWS_EXPORT = True near the top of this file.\n"
            "  3) Implementing build_lightweight_artifact(...).\n"
        )
    # Automated AWS Pipeline
    time.sleep(2)

    random_idx = random.randint(0, len(df)-1)
    customer_data = df.drop("Churn", axis=1).iloc[random_idx].to_dict()

    for k, v in customer_data.items():
        if hasattr(v, "item"): customer_data[k] = v.item()

    my_api_url = "https://your-api-gateway-endpoint.amazonaws.com/prod/telco_churn_predict"

    try:
        response = requests.post(my_api_url, json={"records": [customer_data]})
        print("SUCCESS. The API returned:")
        print(response.json())
    except Exception as e:
        print("Error testing API:", e)

if __name__ == "__main__":
    main()
