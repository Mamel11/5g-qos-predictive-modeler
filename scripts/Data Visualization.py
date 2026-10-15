import os
import json
import datetime
import matplotlib.pyplot as plt
import mysql.connector
import shap
import numpy as np

# Get current working directory
cwd = os.getcwd()
print("Current working directory path:", cwd)

# Construct file paths
graph_path = os.path.join(cwd, "graphs")  # Path to graphs directory

# Database Connection
try:
    mydb = mysql.connector.connect(
        host="localhost",
        user="5gpred",  # Replace with your MySQL username
        password="wyw5445hkwjb@y7^",  # Replace with your MySQL password
        database="5Gpredictivemodeler"
    )

    mycursor = mydb.cursor()

except mysql.connector.Error as err:
    print(f"Database error: {err}")
    exit(1)

# --- Retrieve data for XGBoost and Random Forest ---
model_names = ["XGBoost", "Random Forest"]
results = {}
for model_name in model_names:
    mycursor.execute("SELECT * FROM model_results WHERE model_name = %s", (model_name,))
    result = mycursor.fetchone()
    if result:
        results[model_name] = {
            "timestamp": result[1],
            "best_parameters": json.loads(result[2]),
            "cross_val_rmse": result[3],
            "test_mse": result[4],
            "test_r2": result[5],
            "test_rmse": result[6],
            "all_parameters": json.loads(result[7]),
            "plot_data": json.loads(result[8]),
            "X_test_columns": json.loads(result[9])
        }

# --- 1. Combined SHAP Bar Plot (Replicating Main Code's Style) ---

# Determine the order of features from the database
feature_order = results["XGBoost"]["X_test_columns"] 
# Assuming the order is consistent between models

plt.figure(figsize=(12, 6))

for i, model_name in enumerate(model_names):
    shap_values = np.array(results[model_name]["plot_data"]["shap_bar"])
    base_values = np.array(results[model_name]["plot_data"].get("shap_base_values", 0))
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=results[model_name]["plot_data"]["shap_summary"],
        feature_names=results[model_name]["X_test_columns"]
    )

    # Get the mean absolute SHAP values for sorting
    mean_abs_shap_values = np.abs(shap_explanation.values).mean(0)

    # Sort features based on mean absolute SHAP values
    sorted_features_indices = np.argsort(mean_abs_shap_values)[::-1]  # Descending order
    sorted_features = [results[model_name]["X_test_columns"][idx] for idx in sorted_features_indices]
    sorted_shap_values = shap_explanation.values[:, sorted_features_indices]

    # Convert sorted_features_indices to a list
    sorted_features_indices = list(sorted_features_indices)  # Convert tuple to list

    # Create a new Explanation object with sorted data
    sorted_shap_explanation = shap.Explanation(
        values=sorted_shap_values,
        base_values=shap_explanation.base_values,
        data=shap_explanation.data[:, sorted_features_indices],  # Now uses a list for indexing
        feature_names=sorted_features
    )

    # Plot the bar plot for the current model
    shap.plots.bar(sorted_shap_explanation, show=False)
    plt.title(f"SHAP Feature Importance - {model_name}")

    # Adjust layout for legend
    plt.subplots_adjust(bottom=0.3)

    # Add legend with model details below the plot
    legend_text = (f"Best Hyperparameters: {results[model_name]['best_parameters']}\n"
                   f"Best Cross-Validation Score (RMSE): {results[model_name]['cross_val_rmse']:.2f}\n"
                   f"Training MSE: {results[model_name]['test_mse']:.2f}\n"
                   f"Training R^2: {results[model_name]['test_r2']:.2f}\n"
                   f"RMSE: {results[model_name]['test_rmse']:.2f}")
    plt.figtext(0.5, 0.01, legend_text, ha="center", fontsize=10,
                bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5})

    plt.savefig(os.path.join(graph_path, f"combined_shap_bar_plot_{model_name}_{timestamp}.png"), dpi=300)
    plt.show()

# --- 2. Combined SHAP Summary Plot ---

plt.figure(figsize=(12, 6))

# Define colors for each model (blue for XGBoost, red for Random Forest)
model_colors = ["blue", "red"]

# Get the list of features (assuming both models have the same features)
features = results["XGBoost"]["X_test_columns"]

# Calculate the required height for the plot based on the number of features and models
plot_height = 0.8 + (len(features) * len(model_names) * 0.2)  # Adjusted height calculation

# Adjust the figure height accordingly
plt.figure(figsize=(12, plot_height))

for i, model_name in enumerate(model_names):
    shap_values = np.array(results[model_name]["plot_data"]["shap_bar"])
    base_values = np.array(results[model_name]["plot_data"].get("shap_base_values", 0))
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=results[model_name]["plot_data"]["shap_summary"],
        feature_names=features
    )

    # Create the SHAP summary plot with the specified color
    shap.summary_plot(shap_explanation, show=False, color=model_colors[i], sort=False)

plt.title("SHAP Summary Plot - XGBoost VS. Random Forest")  # Updated title

# Adjust layout for legend
plt.subplots_adjust(bottom=0.2)  # Adjusted bottom margin

# Add legend with model details
plt.figtext(0.5, 0.01, "\n".join(legend_text), ha="center", fontsize=10,
            bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5})

plt.savefig(os.path.join(graph_path, f"combined_shap_summary_plot_{timestamp}.png"), dpi=300)
plt.show()

# --- 3. Model Performance Comparison Plot ---
plt.figure(figsize=(10, 6))
metrics = ["cross_val_rmse", "test_mse", "test_r2", "test_rmse"]
bar_width = 0.35
index = np.arange(len(metrics))

for i, model_name in enumerate(model_names):
    plt.bar(index + i * bar_width, [results[model_name][metric] for metric in metrics], 
            bar_width, label=model_name)

plt.xlabel("Metrics")
plt.ylabel("Values")
plt.title("Model Performance Comparison - XGBoost vs. Random Forest")
plt.xticks(index + bar_width / 2, metrics)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(graph_path, f"model_performance_comparison_{timestamp}.png"), dpi=300)
plt.show()

