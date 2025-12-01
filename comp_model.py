import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RidgeCV, LassoCV, HuberRegressor, LinearRegression, SGDRegressor

FEATURE_GROUPS = {
    "GDP": ["GDMPBC"],
    "CPI": ["All Items Index", "Annual Percent Change"],
    "Unemployment": [
        'Total, all education levels',
        '0 to 8  years 8',
        'Some high school 9',
        'High school graduate 10',
        'Some postsecondary 11',
        'Postsecondary certificate or diploma 12',
        'University degree 13',
        "Bachelor's degree",
        'Above bachelor\'s degree'
    ]
}

MODEL_MAP = {
    "OLS": lambda: LinearRegression(),
    "Ridge": lambda: RidgeCV(cv=5),
    "Lasso": lambda: LassoCV(cv=5, alphas=np.logspace(-5, 0, 100), max_iter=100000),
    "Huber": lambda: HuberRegressor(max_iter=100000, alpha=1e-7, epsilon=1),
    "SGD": lambda: SGDRegressor(penalty='l2', max_iter=100000000, alpha=0.001, early_stopping=True)
}



def LR_train(model, X_train, Y_train, X_test, Y_test):
    LR_model = model
    LR_model.fit(X_train, Y_train)

    y_pred = LR_model.predict(X_test)
    mse = mean_squared_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)

    print(f'[Trend] Mean Squared Error: {mse:.2f}, Sqrt MSE: {np.sqrt(mse):.2f}')
    print(f'[Trend] R^2 Score: {r2:.2f}')

    return LR_model

def plot_results(df_test, trend_pred_test, final_pred, info, args):
    df_plot = df_test.copy()
    df_plot.loc[:, 'YEAR'] = df_plot['YEAR'] + (df_plot['QUARTER'] - 1) * 0.25
    
    plt.figure(figsize=(12, 6))
    plt.scatter(df_plot['YEAR'], df_plot['MEDIAN HOUSING PRICE'], color='blue', label='Actual', s=50)
    plt.plot(df_plot['YEAR'], final_pred, color='red', linewidth=2, label='Trend + Tree')
    plt.plot(df_plot['YEAR'], trend_pred_test, color='orange', linewidth=2, linestyle='--', label='Trend Only')
    
    plt.xlabel('Year')
    plt.ylabel('Median Housing Price ($CAD)')
    plt.title(f'Housing Price Prediction: {info}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"Output/CM-{args.model}-{args.splityear}-{args.featurex}-{args.disable_boosting}")
    plt.show()



def main():
    parser = argparse.ArgumentParser(description="Run linear model experiment")
    parser.add_argument("-m", "--model",
                        type=str, default="Lasso",
                        choices=["OLS", "Ridge", "Lasso", "Huber", "SGD"],
                        help="Choose linear model")
    parser.add_argument("-s", "--splityear",
                        type=int, default=2018,
                        help="Choose year to split dataset on: range(1991, 2022)")
    parser.add_argument("-fx", "--featurex",
                        type=str, default="None",
                        choices=["None", "GDP", "CPI", "Unemployment"],
                        help="Optional feature group to exclude")
    parser.add_argument("--disable-boosting",
                        action="store_true",
                        help="Disable Gradient Tree Boosting")
    args = parser.parse_args()

    info = f"Model: {args.model}, Split on Year: {args.splityear}, Excluded Feature Group: {args.featurex}, Boosting: {not args.disable_boosting}"
    print(info)

    # remove excluded features
    rem_features = FEATURE_GROUPS.get(args.featurex, [])
    trend_features = ['YEAR', 'QUARTER', 'GDMPBC', 'All Items Index', 'Annual Percent Change', 'MEDIAN AFTER TAX FAMILY INCOME',
                      'PRIME INTEREST RATE','QUARTERLY PAYMENT','PAYMENT TO INCOME PERCENT','NUMBER OF RESALES']
    tree_features = ['Total, all education levels', 'High school graduate 10', 'Some high school 9', 'Some postsecondary 11', 
                     'Postsecondary certificate or diploma 12', "Bachelor's degree", '0 to 8  years 8', 'University degree 13', "Above bachelor's degree", 
                     'PRIME INTEREST RATE', 'MEDIAN AFTER TAX FAMILY INCOME']
    
    utrend_features = list(set(trend_features) - (set(rem_features) & set(trend_features)))
    utree_features = list(set(tree_features) - (set(rem_features) & set(tree_features)))

    df = pd.read_csv('Data/edited data/merged_data.csv')
    df = df.astype(float)
    df_train = df[df['YEAR'] <= args.splityear]
    df_test = df[df['YEAR'] > args.splityear]

    # LR train & eval
    X_train = df_train.drop(columns=["MEDIAN HOUSING PRICE"])[utrend_features]
    Y_train = df_train["MEDIAN HOUSING PRICE"]
    X_test = df_test.drop(columns=["MEDIAN HOUSING PRICE"])[utrend_features]
    Y_test = df_test["MEDIAN HOUSING PRICE"]
    
    LR_pipeline = make_pipeline(StandardScaler(), 
                                MODEL_MAP[args.model]())
    LR_model = LR_train(model=LR_pipeline, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

    trend_pred_train = LR_model.predict(X_train)
    trend_pred_test = LR_model.predict(X_test)

    # Residual Boosting Technique (Decision Tree)
    residuals = df_train['MEDIAN HOUSING PRICE'] - trend_pred_train
    tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    tree_model.fit(df_train[utree_features], residuals)
    residual_pred = tree_model.predict(df_test[utree_features])

    # Final prediction by combining trend and residuals
    final_pred = trend_pred_test + residual_pred

    # Evaluate combined model
    mse = mean_squared_error(df_test['MEDIAN HOUSING PRICE'], final_pred)
    r2 = r2_score(df_test['MEDIAN HOUSING PRICE'], final_pred)
    print(f'[Trend + Tree] Mean Squared Error: {mse:.2f}, Sqrt MSE: {np.sqrt(mse):.2f}')
    print(f'[Trend + Tree] R^2 Score: {r2:.2f}')

    plot_results(df_test, trend_pred_test, final_pred, info, args)



if __name__ == '__main__':
    main()