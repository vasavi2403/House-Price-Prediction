# House-Price-Prediction

A beginner machine learning project that predicts house sale prices using the [Kaggle House Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). Built with Python and scikit-learn.

---

## Results

| Model | Mean Absolute Error | R² Score |
|---|---|---|
| Linear Regression | $21,411 | 0.847 |
| **Random Forest** | **$17,450** | **0.898** |

The Random Forest model predicts house prices to within ~$17K on average. The most influential feature by far is **Overall Quality rating**, followed by **above-ground living area (sqft)**.

---

## Project Structure

```
house-price-prediction/
├── train.csv                    # Kaggle dataset (download separately)
├── kaggle_house_prices.py       # Main script
├── kaggle_results.png           # Output charts
└── README.md
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### 2. Install dependencies

```bash
pip install pandas scikit-learn matplotlib numpy
```

### 3. Download the dataset

1. Go to [kaggle.com/c/house-prices-advanced-regression-techniques/data](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
2. Sign in and click **Download All**
3. Unzip and place `train.csv` in the project folder

### 4. Run the script

```bash
python kaggle_house_prices.py
```

---

## What the Script Does

1. **Loads** the dataset (1,460 houses, 81 columns)
2. **Cleans** the data — drops columns with >50% missing values, fills the rest with medians or most-common values
3. **Encodes** categorical (text) columns into numbers using `LabelEncoder`
4. **Splits** data into 80% training / 20% test
5. **Trains** two models: Linear Regression and Random Forest
6. **Evaluates** using Mean Absolute Error (MAE) and R²
7. **Visualises** results with 4 charts saved as `kaggle_results.png`

---

## Output Charts

- **Actual vs Predicted** — how close predictions are to real prices (for both models)
- **Top 10 Feature Importances** — which house features drive the price most
- **Price Distribution** — histogram of sale prices in the dataset

---

## Key Concepts Learned

- Handling missing values (dropping vs filling)
- Encoding categorical data for ML models
- Train/test splitting
- Comparing Linear Regression vs Random Forest
- Feature importance from ensemble models
- Evaluating regression models with MAE and R²

---

## Ideas to Extend This

- [ ] Add feature engineering (e.g. `TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF`)
- [ ] Try `n_estimators=200` in Random Forest for better accuracy
- [ ] Use `GridSearchCV` to tune hyperparameters
- [ ] Generate predictions on `test.csv` and submit to the Kaggle leaderboard
- [ ] Try `XGBoost` — currently one of the top performers on this dataset

---

## Dependencies

- Python 3.8+
- pandas
- scikit-learn
- matplotlib
- numpy

---

## Dataset

**Kaggle — House Prices: Advanced Regression Techniques**  
1,460 residential homes in Ames, Iowa with 79 features describing almost every aspect of the property.  
→ [kaggle.com/c/house-prices-advanced-regression-techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## Author

Built as a beginner ML learning project.
