import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error

def main():
    df = pd.read_csv('Data/edited data/merged_data.csv')
    df = df.astype(float)

    df_filt_train = df[df['YEAR'] <= 2018]
    df_filt_test = df[df['YEAR'] > 2018]

    X_train = df_filt_train.drop(columns=["MEDIAN HOUSING PRICE"])
    Y_train = df_filt_train["MEDIAN HOUSING PRICE"]
    X_test = df_filt_test.drop(columns=["MEDIAN HOUSING PRICE"])
    Y_test = df_filt_test["MEDIAN HOUSING PRICE"]

    scalerX = StandardScaler()
    X_train_scaled = scalerX.fit_transform(X_train)
    X_test_scaled = scalerX.transform(X_test)

    ridge_cv = LassoCV(cv=5)
    ridge_cv.fit(X_train_scaled, Y_train)

    y_pred = ridge_cv.predict(X_test_scaled)
    mse = mean_squared_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)

    coef_label = pd.DataFrame({
        'Feats': X_test.columns,
        'Coefs': ridge_cv.coef_
    })
    print(coef_label.sort_values(by='Coefs'))
    print(f'Mean Squared Error: {mse:.2f}, Sqrt MSE: {np.sqrt(mse):.2f}')
    print(f'R^2 Score: {r2:.2f}')

if __name__ == '__main__':
    main()