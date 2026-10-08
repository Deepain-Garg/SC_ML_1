# House Price Prediction using Linear Regression

This project implements a Multiple Linear Regression model from scratch (using `scikit-learn` for regression modeling) to predict house prices based on three features: **Square Footage**, **Number of Bedrooms**, and **Number of Bathrooms**.

The dataset consists of 101 data points capturing real estate prices (in Lakhs) across various configurations.

---

## Project Structure

```text
ml/
├── house_prices.csv        # Tabular dataset containing 101 records
├── train.py                # Model training, evaluation, and plot generation script
├── predict.py              # CLI/Interactive prediction utility
├── house_price_model.pkl   # Serialized trained Linear Regression model
└── visualizations/         # Saved plots and figures
    ├── correlation_matrix.png
    └── predictions_vs_actual.png
```

---

## Dataset Description

The dataset `house_prices.csv` contains the following columns:
1. **`SquareFeet`**: The total floor area of the house (numerical, ranging from 850 to 11,000 sq ft).
2. **`Bedrooms`**: The number of bedrooms (integer, ranging from 1 to 8).
3. **`Bathrooms`**: The number of bathrooms (integer, ranging from 1 to 7).
4. **`Price`**: The target house value in **Lakhs** (numerical, ranging from 32 to 880 Lakhs).

### Correlation Matrix Analysis
The features are highly correlated with one another, especially `SquareFeet` with the target `Price`. The correlation matrix heatmap is saved automatically under [visualizations/correlation_matrix.png](file:///c:/Users/USER/Desktop/ml/visualizations/correlation_matrix.png) during training.

---

## Requirements and Installation

This project requires **Python 3.8+** (verified on Python 3.13.9) and the following standard scientific computing packages:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

To install the dependencies, run:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Model Training & Performance

To train the linear regression model and generate metrics/visualizations, execute:
```bash
python train.py
```

### Regression Model Formula
After fitting the model on an 80/20 train/test split, the trained Multiple Linear Regression model has the following equation:

$$\text{Price (Lakhs)} = -18.3167 + (0.0940 \times \text{SquareFeet}) - (9.4306 \times \text{Bedrooms}) - (12.8753 \times \text{Bathrooms})$$

#### Interpretation of Coefficients:
- **`SquareFeet` ($+0.0940$)**: For every additional square foot, the house price increases by approximately $0.0940$ Lakhs ($9,400$ units of currency) assuming other features are held constant.
- **`Bedrooms` ($-9.4306$)** & **`Bathrooms` ($-12.8753$)**: The negative coefficients for bedrooms and bathrooms are a classic symptom of **multicollinearity**. Because larger square footage houses naturally have more bedrooms and bathrooms, these three features are heavily correlated. In a multiple linear regression model, the negative coefficients indicate that *for a fixed square footage*, adding more bedrooms or bathrooms reduces the expected price (likely because it divides the house into smaller, less premium rooms, or represents a specific layout trade-off in the dataset).

### Evaluation Metrics (Test Set)
The model exhibits exceptional fit quality:
- **Mean Absolute Error (MAE)**: 7.1040 Lakhs
- **Mean Squared Error (MSE)**: 76.2276 (Lakhs²)
- **Root Mean Squared Error (RMSE)**: 8.7308 Lakhs
- **R-squared ($R^2$ Score)**: 0.997773 (explains **99.77%** of the variance in house prices)

---

## Running Predictions

You can make predictions using the trained model in two ways:

### 1. Command Line Interface (CLI) Mode
Pass the parameters directly as arguments:
```bash
python predict.py --square_feet 2500 --bedrooms 4 --bathrooms 3
```

**Example Output:**
```text
==================================================
                 PREDICTION RESULTS                 
==================================================
  Inputs:
    - Square Footage:   2,500.00 sq ft
    - Bedrooms:         4.0
    - Bathrooms:        3.0
--------------------------------------------------
  Predicted House Price: 140.2628 Lakhs
==================================================
```

### 2. Interactive Mode
Run the script without any arguments, and you will be prompted for input:
```bash
python predict.py
```

**Example Prompts:**
```text
--- Enter House Details for Prediction ---
Enter Square Footage (e.g., 2500): 1850
Enter Number of Bedrooms (e.g., 4): 3
Enter Number of Bathrooms (e.g., 3): 3

==================================================
                 PREDICTION RESULTS                 
==================================================
  Inputs:
    - Square Footage:   1,850.00 sq ft
    - Bedrooms:         3.0
    - Bathrooms:        3.0
--------------------------------------------------
  Predicted House Price: 88.6644 Lakhs
==================================================
```

---

## Visualizations

The training process generates two high-resolution plots under the `visualizations/` folder:

1. **[correlation_matrix.png](file:///c:/Users/USER/Desktop/ml/visualizations/correlation_matrix.png)**: A heatmap showcasing Pearson correlation coefficients.
2. **[predictions_vs_actual.png](file:///c:/Users/USER/Desktop/ml/visualizations/predictions_vs_actual.png)**: A scatter plot comparing predicted prices vs actual prices on the test dataset. A dashed line represents a perfect $y = x$ fit line for visual comparison.
