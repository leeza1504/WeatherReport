import pandas as pd
weather = pd.read_csv("weather.csv", index_col="DATE")
weather
# print (weather) 
# null_pct = weather.apply(pd.isnull).sum()/weather.shape[0]
# # print (null_pct)
# valid_columns = weather.columns[null_pct < .05]
# # print (valid_columns)
# weather = weather[valid_columns].copy()
# weather.columns = weather.columns.str.lower()
# # print (weather)
# weather = weather.ffill()
# print (weather.apply(pd.isnull).sum())
# # print (weather) 
# weather.dtypes
# print (weather.dtypes)
null_pct = weather.apply(pd.isnull).sum()/weather.shape[0]
valid_columns = weather.columns[null_pct < .05]
weather = weather[valid_columns].copy()
weather.columns = weather.columns.str.lower()
weather = weather.ffill()
weather.index = pd.to_datetime(weather.index)

weather["target"] = weather.shift(-1) ["tmax"]

weather = weather.ffill()

from sklearn.linear_model import Ridge
print(weather.corr(numeric_only=True))
rr=Ridge(alpha=.1)
predictor=weather.columns[~weather.columns.isin(["target","name","station"])]
# print(predictor)


def backtest(weather, model, predictors, start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i,:]
        test = weather.iloc[i:(i+step),:]

        model.fit(train[predictors], train["target"])

        preds = model.predict(test[predictors])

        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)

        combined.columns = ["actual", "prediction"]

        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()

        all_predictions.append(combined)
    return pd.concat(all_predictions)
predictions = backtest(weather, rr, predictor)
print(predictions)

