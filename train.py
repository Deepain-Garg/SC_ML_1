import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Define paths
DATA_PATH = "house_prices.csv"
MODEL_PATH = "house_price_model.pkl"
VIS_DIR = "visualizations"

def main():
    print("=" * 60)
    print("           HOUSE PRICE PREDICTION - MODEL TRAINING          ")
    print("=" * 60)

    # 1. Load the dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset file '{DATA_PATH}' not found. Please create it first.")
    
    print(f"[*] Loading dataset from '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH)
    print(f"[+] Loaded {len(df)} samples with features: {list(df.columns)}")
    print("\nDataset Summary Statistics:")
    print(df.describe().to_string())
    print("-" * 60)

    # 2. Setup visualization directory
    os.makedirs(VIS_DIR, exist_ok=True)
    print(f"[*] Created output directory for plots: '{VIS_DIR}/'")

    # 3. Generate and save correlation matrix heatmap
    print("[*] Generating correlation matrix heatmap...")
    plt.figure(figsize=(8, 6), dpi=150)
    sns.set_theme(style="whitegrid")
    
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap="coolwarm", 
        mask=mask,
        vmin=-1, 
        vmax=1, 
        fmt=".3f", 
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Feature Correlation Matrix Heatmap", fontsize=14, pad=20, weight='bold')
    plt.tight_layout()
    corr_plot_path = os.path.join(VIS_DIR, "correlation_matrix.png")
    plt.savefig(corr_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved correlation heatmap to '{corr_plot_path}'")

    # 4. Prepare data for training
    X = df[['SquareFeet', 'Bedrooms', 'Bathrooms']]
    y = df['Price']

    # Split into train and test sets (80% train, 20% test)
    # Using random_state=42 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[+] Split data: {len(X_train)} training samples, {len(X_test)} test samples.")

    # 5. Train Linear Regression Model
    print("[*] Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("[+] Model training complete.")

    # Get coefficients and intercept
    intercept = model.intercept_
    coef_sqft, coef_beds, coef_baths = model.coef_

    print("\nTrained Model Equation:")
    print(f"  Price (Lakhs) = {intercept:.4f} \n"
          f"                  + ({coef_sqft:.4f} * SquareFeet) \n"
          f"                  + ({coef_beds:.4f} * Bedrooms) \n"
          f"                  + ({coef_baths:.4f} * Bathrooms)")
    
    print("\nFeature Importances (Coefficients):")
    print(f"  - SquareFeet: {coef_sqft:.4f} Lakhs per sq ft")
    print(f"  - Bedrooms:   {coef_beds:.4f} Lakhs per bedroom")
    print(f"  - Bathrooms:  {coef_baths:.4f} Lakhs per bathroom")
    print("-" * 60)

    # 6. Evaluate on Test Set
    print("[*] Evaluating model on the test set...")
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nTest Performance Metrics:")
    print(f"  - Mean Absolute Error (MAE):       {mae:.4f} Lakhs")
    print(f"  - Mean Squared Error (MSE):        {mse:.4f} (Lakhs^2)")
    print(f"  - Root Mean Squared Error (RMSE):  {rmse:.4f} Lakhs")
    print(f"  - R-squared (R2 Score):            {r2:.6f} ({r2*100:.4f}%)")
    print("-" * 60)

    # 7. Generate and save Actual vs Predicted scatter plot
    print("[*] Generating actual vs. predicted plot...")
    plt.figure(figsize=(8, 6), dpi=150)
    
    # Plot data points
    plt.scatter(y_test, y_pred, color="#1f77b4", alpha=0.7, edgecolors='w', s=80, label="Test Predictions")
    
    # Plot reference line y = x
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#d62728", linestyle="--", linewidth=2, label="Perfect Predictor (y = x)")
    
    plt.xlabel("Actual Price (Lakhs)", fontsize=12, labelpad=10)
    plt.ylabel("Predicted Price (Lakhs)", fontsize=12, labelpad=10)
    plt.title("Actual vs. Predicted House Prices", fontsize=14, pad=15, weight='bold')
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    pred_plot_path = os.path.join(VIS_DIR, "predictions_vs_actual.png")
    plt.savefig(pred_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved prediction plot to '{pred_plot_path}'")

    # 8. Save the model
    print(f"[*] Saving trained model to '{MODEL_PATH}'...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print("[+] Model saved successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
