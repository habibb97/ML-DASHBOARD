from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib

matplotlib.use("Agg")  # Gunakan backend non-GUI
import matplotlib.pyplot as plt
import io
import base64
import time

app = Flask(__name__)


# Generate synthetic dataset
def generate_data():
    np.random.seed(42)
    X = np.random.rand(1000, 5)
    y = X @ np.array([3, -2, 1, 0.5, 0.1]) + np.random.normal(0, 0.1, 1000)
    return train_test_split(X, y, test_size=0.3, random_state=42)


# Convert plot to base64 string
def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode("utf8")
    buf.close()
    plt.close(fig)
    return base64_img


# Plot Residuals
def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")
    return fig


# Plot Error Histogram
def plot_error_histogram(y_test, y_pred):
    errors = y_test - y_pred
    fig, ax = plt.subplots()
    ax.hist(errors, bins=20, color="skyblue", edgecolor="black")
    ax.set_title("Error Histogram")
    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    return fig


# Plot Feature Importance
def plot_feature_importance(importances):
    fig, ax = plt.subplots()
    ax.bar(range(len(importances)), importances)
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importance")
    return fig


# Plot Prediction vs Actual
def plot_prediction_vs_actual(y_test, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6, color="blue", edgecolors="black")
    ax.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        color="red",
        linestyle="--",
        linewidth=1,
    )
    ax.set_title("Prediction vs Actual")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    return fig


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train_model():
    model_type = request.args.get("model", "linear")

    # Load data
    X_train, X_test, y_train, y_test = generate_data()

    # Initialize model
    if model_type == "ridge":
        model = Ridge()
    elif model_type == "lasso":
        model = Lasso()
    elif model_type == "random_forest":
        model = RandomForestRegressor()
    else:
        model = LinearRegression()

    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Generate plots
    residual_plot = plot_to_base64(plot_residuals(y_test, y_pred))
    error_histogram_plot = plot_to_base64(plot_error_histogram(y_test, y_pred))
    prediction_vs_actual_plot = plot_to_base64(
        plot_prediction_vs_actual(y_test, y_pred)
    )
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        feature_importances = np.abs(model.coef_)
    else:
        feature_importances = np.zeros(X_train.shape[1])
    feature_importance_plot = plot_to_base64(
        plot_feature_importance(feature_importances)
    )

    response = {
        "metrics": {
            "accuracy": f"{r2 * 100:.2f}%",
            "mse": mse,
            "training_time": f"{training_time:.2f} seconds",
            "data_points": len(X_train) + len(X_test),
        },
        "plots": {
            "residual_plot": residual_plot,
            "error_histogram": error_histogram_plot,
            "prediction_vs_actual": prediction_vs_actual_plot,
            "feature_importance": feature_importance_plot,
        },
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
