 # CLEAN & PROFESSIONAL VERSION


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. Load Dataset

df = pd.read_csv("gemstone.csv")
print("\nDataset Loaded")
print(df.head())
print(df.describe())


# 2. Visual Exploratory Data Analysis

plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.hist(df['price'], bins=50)
plt.title("Price Distribution")

plt.subplot(2, 2, 2)
plt.scatter(df['carat'], df['price'], alpha=0.5)
plt.title("Carat vs Price")
plt.xlabel("Carat")
plt.ylabel("Price")

plt.subplot(2, 2, 3)
sns.barplot(x="cut", y="price", data=df)
plt.title("Cut vs Average Price")

plt.subplot(2, 2, 4)
sns.heatmap(df[['carat','depth','table','x','y','z','price']].corr(),
            annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# 3. Prepare Data

df = df.drop("id", axis=1)

X = df.drop("price", axis=1)
y = df["price"]

categorical_cols = ['cut', 'color', 'clarity']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train CatBoost Model

model = CatBoostRegressor(
    iterations=800,
    learning_rate=0.1,
    depth=6,
    cat_features=categorical_cols,
    loss_function='RMSE',
    random_seed=42,
    verbose=False
)

model.fit(X_train, y_train, eval_set=(X_test, y_test))

# 5. Evaluate Model

y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\nModel Performance")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# 6. Feature Importance

importances = model.get_feature_importance()
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()


# 7. Prediction Accuracy Visuals

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")

plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.hist(residuals, bins=40)
plt.title("Residual Distribution")
plt.xlabel("Error")

plt.tight_layout()
plt.show()


# 8. Predict New Gemstones

new_data = pd.DataFrame({
    'carat':   [1.2, 0.9],
    'cut':     ['Ideal', 'Good'],
    'color':   ['F', 'H'],
    'clarity': ['VS1', 'SI1'],
    'depth':   [61.5, 62.0],
    'table':   [57, 58],
    'x':       [6.8, 5.4],
    'y':       [6.9, 5.3],
    'z':       [4.2, 3.4]
})

new_predictions = model.predict(new_data)
print("\nPredicted Prices for New Stones:")
print(new_predictions)
