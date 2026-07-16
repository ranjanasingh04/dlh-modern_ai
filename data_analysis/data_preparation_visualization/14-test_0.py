#!/usr/bin/env python3

import pandas as pd
create_features = __import__('14-create_features').create_features
plot_numeric_vs_churn = __import__('11-plot_numeric_vs_churn').plot_numeric_vs_churn


df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
df.drop(columns=['gender', 'PhoneService'], inplace=True)
df = create_features(df)
plot_numeric_vs_churn(df, 'NumServices')
