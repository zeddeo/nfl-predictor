from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def train_and_predict(df, season, week, features=['Spread', 'Total']):
    train_df = df.query('Season < @season or (Season == @season and Week < @week)')
    test_df = df.query('Season == @season and Week == @week and True_Total != Total')

    if test_df.empty:
        return None, None, 0.0, 0.0

    X_train = train_df[features]
    y_under_train = train_df['Under']
    y_win_train = train_df['Rslt']

    X_test = test_df[features]
    y_under_test = test_df['Under']
    y_win_test = test_df['Rslt']

    model_under = KNeighborsClassifier(n_neighbors=7)
    model_win = KNeighborsClassifier(n_neighbors=7)

    model_under.fit(X_train, y_under_train)
    model_win.fit(X_train, y_win_train)

    under_pred = model_under.predict(X_test)
    win_pred = model_win.predict(X_test)

    acc_under = accuracy_score(y_under_test, under_pred)
    acc_win = accuracy_score(y_win_test, win_pred)

    return under_pred.tolist(), win_pred.tolist(), acc_under, acc_win
