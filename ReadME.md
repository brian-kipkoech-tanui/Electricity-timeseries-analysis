Sourced from kaggle
## Dataset Description
In this competition, you will predict sales for the thousands of product families sold at Favorita stores located in Ecuador. The training data includes dates, store and product information, whether that item was being promoted, as well as the sales numbers. Additional files include supplementary information that may be useful in building your models.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

For the ith sample, Squared Logarithmic Error is calculated as SLE = (log(prediction + 1) - log(actual + 1))^2. RMSLE is then sqrt(mean(squared logarithmic errors)).


```python
# create the .kaggle directory and an empty kaggle.json file
!mkdir -p C:\Users\HP
!touch C:\Users\HP\.kaggle/kaggle.json
!chmod 600 C:\Users\HP\.kaggle\kaggle.json
```

    A subdirectory or file -p already exists.
    Error occurred while processing: -p.
    A subdirectory or file C:\Users\HP already exists.
    Error occurred while processing: C:\Users\HP.
    


```python
# Fill in your user name and key from creating the kaggle account and API token file
import json
kaggle_username = "tanui2019"
kaggle_key = "bd1772ca85673cd91912c0897c40d703"

# Save API token the kaggle.json file
with open("C:/Users/HP/.kaggle/kaggle.json", "w") as f:
    f.write(json.dumps({"username": kaggle_username, "key": kaggle_key}))
```


```python
# Download the time series dataset
# !kaggle competitions download -c store-sales-time-series-forecasting
```


```python
# Unzip the files
!unzip -o store-sales-time-series-forecasting.zip # Will not work
```

    Archive:  store-sales-time-series-forecasting.zip
    

    caution: filename not matched:  #
    caution: filename not matched:  Will
    caution: filename not matched:  not
    caution: filename not matched:  work
    


```python
import zipfile

# Specify the path to the ZIP file
zip_file_path = '../data/store-sales-time-series-forecasting.zip'

# Specify the directory where you want to extract the contents
extracted_dir = './'

# Open the ZIP file and extract its contents
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)
```


```python
train_csv = pd.read_csv('../data/train.csv')
test_csv = pd.read_csv('../data/test.csv')
transactions_csv = pd.read_csv('../data/transactions.csv')
holidays_csv = pd.read_csv('../data/holidays_events.csv')
```


```python
holidays_csv.drop(["description","locale_name","locale_name"], axis=1, inplace=True)
```


```python
holidays_csv.drop_duplicates(inplace=True)
```


```python
holidays_type_csv = holidays_csv.groupby(["date"], as_index=False)["type"].first()
holidays_locale_csv = holidays_csv.groupby(["date"], as_index=False)["locale"].first()
```


```python
train_csv.isna().sum()
```




    id             0
    date           0
    store_nbr      0
    family         0
    sales          0
    onpromotion    0
    dtype: int64




```python
train_csv['sales'].sum()
```




    1073644952.2030689




```python
train_csv = train_csv.merge(holidays_type_csv, how='left', on='date')
test_csv = test_csv.merge(holidays_type_csv, how='left', on='date')

train_csv = train_csv.merge(holidays_locale_csv, how='left', on='date')
test_csv = test_csv.merge(holidays_locale_csv, how='left', on='date')
```


```python
test_csv
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>store_nbr</th>
      <th>family</th>
      <th>onpromotion</th>
      <th>type</th>
      <th>locale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3000888</td>
      <td>2017-08-16</td>
      <td>1</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3000889</td>
      <td>2017-08-16</td>
      <td>1</td>
      <td>BABY CARE</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3000890</td>
      <td>2017-08-16</td>
      <td>1</td>
      <td>BEAUTY</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3000891</td>
      <td>2017-08-16</td>
      <td>1</td>
      <td>BEVERAGES</td>
      <td>20</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3000892</td>
      <td>2017-08-16</td>
      <td>1</td>
      <td>BOOKS</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>28507</th>
      <td>3029395</td>
      <td>2017-08-31</td>
      <td>9</td>
      <td>POULTRY</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28508</th>
      <td>3029396</td>
      <td>2017-08-31</td>
      <td>9</td>
      <td>PREPARED FOODS</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28509</th>
      <td>3029397</td>
      <td>2017-08-31</td>
      <td>9</td>
      <td>PRODUCE</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28510</th>
      <td>3029398</td>
      <td>2017-08-31</td>
      <td>9</td>
      <td>SCHOOL AND OFFICE SUPPLIES</td>
      <td>9</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28511</th>
      <td>3029399</td>
      <td>2017-08-31</td>
      <td>9</td>
      <td>SEAFOOD</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>28512 rows × 7 columns</p>
</div>




```python
def fix_missing_values(data):
    # Fill the numeric missing values with median
    numeric_cols = data.select_dtypes(include=["int", "float"]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    # Fill the string missing values with mode
    string_cols = data.select_dtypes(include=["object"]).columns
    for col in string_cols:  # Get the first mode from the Series
        data[col] = data[col].fillna("Not Applicable")

    return data

train_csv = fix_missing_values(train_csv)
test_csv = fix_missing_values(test_csv)
```

## Analyze the general Sales data without other predictors


```python
sales = train_csv[["date", "sales"]]
sales = sales.groupby("date", as_index=False)["sales"].sum()
sales["date"] = pd.to_datetime(sales['date'])
```


```python
sales["sales"].sum()
```




    1073644952.2030686




```python
import os
os.chdir('./TimeSeriesclass/')
```


```python
from basicTimeSeries import SARIMAModels
ts = SARIMAModels()
```


```python
sales.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1684 entries, 0 to 1683
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype         
    ---  ------  --------------  -----         
     0   date    1684 non-null   datetime64[ns]
     1   sales   1684 non-null   float64       
    dtypes: datetime64[ns](1), float64(1)
    memory usage: 26.4 KB
    


```python
sales.set_index('date', inplace=True)
```


```python
# Resample the data to daily frequency and fill missing values
sales = sales.resample('D').asfreq()
```


```python
sales = sales.fillna(method='ffill')
```


```python
sales.isna().sum()
```




    sales    0
    dtype: int64




```python
ts.trend_visualizations(sales)
```


    
![png](output_27_0.png)
    



```python
fig, ax = plt.subplots()
sales.plot(ax=ax)
sales.rolling(10).mean().plot(ax=ax, c='red') # 10-point rolling mean
sales.rolling(20).mean().plot(ax=ax, c='midnightblue') # 20-point rlling mean
plt.legend(["Time Plot", "10-point MA", "20-point MA"])
plt.show()
```


    
![png](output_28_0.png)
    



```python
ts.decomposition_plot(sales)
```


    
![png](output_29_0.png)
    



    
![png](output_29_1.png)
    



    
![png](output_29_2.png)
    



```python
ts.correlation_function(sales, lags=100)
```


    
![png](output_30_0.png)
    



    
![png](output_30_1.png)
    



```python
ts.stationarity_check(sales)
```

        Augmented Dickey-Fuller Test on "sales" 
        -----------------------------------------------
     Null Hypothesis: Data has unit root. Non-Stationary.
     Significance Level    = 0.05
     Test Statistic        = -2.6233
     No. Lags Chosen       = 22
     Critical value 1%     = -3.434
     Critical value 5%     = -2.863
     Critical value 10%    = -2.568
     => P-Value = 0.0883. Weak evidence to reject the Null Hypothesis.
     => Series is Non-Stationary.
    
    
    


```python
train, test = ts.split_data(sales, ratio=0.8)
```


```python
# ts.model_evaluation(train_data=train, order_limit=2)
```


```python
residuals = ts.best_model(train, test, order=(2, 1, 1),
                         seasonal_order=(2, 1, 1, 12), lags=1)
```

                                         SARIMAX Results                                      
    ==========================================================================================
    Dep. Variable:                              sales   No. Observations:                 1350
    Model:             SARIMAX(2, 1, 1)x(2, 1, 1, 12)   Log Likelihood              -17435.659
    Date:                            Sun, 22 Oct 2023   AIC                          34885.319
    Time:                                    15:23:12   BIC                          34921.569
    Sample:                                01-01-2013   HQIC                         34898.914
                                         - 09-11-2016                                         
    Covariance Type:                              opg                                         
    ==========================================================================================
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1          0.4990      0.069      7.259      0.000       0.364       0.634
    ar.L2         -0.2925      0.075     -3.895      0.000      -0.440      -0.145
    ma.L1         -0.8566      0.060    -14.277      0.000      -0.974      -0.739
    ar.S.L12      -0.3146      0.064     -4.914      0.000      -0.440      -0.189
    ar.S.L24      -0.1773      0.074     -2.381      0.017      -0.323      -0.031
    ma.S.L12      -1.0010      0.012    -80.575      0.000      -1.025      -0.977
    sigma2      4.131e+10   2.85e-13   1.45e+23      0.000    4.13e+10    4.13e+10
    ==============================================================================
    
    
    ----------------------------------------------- 
     ----------------------------------------------- 
    
    Model Diagnostics
    


    
![png](output_34_1.png)
    


    
    
    ----------------------------------------------- 
     ----------------------------------------------- 
    
    Ljung-Box Test - Checks for Serial Correlation 
    
    null-hypothesis:         The residuals are independently distributed.
    alternative-hypothesis:  The residuals are not independently distributed 
     -----------------------------------------------
        lb_stat  lb_pvalue
    1  0.111833   0.738067
    
    
    ----------------------------------------------- 
     ----------------------------------------------- 
    
    How predictions compares to actual test data 
    
    


    
![png](output_34_3.png)
    



```python
predictions = ts.forecasts(sales["sales"],
                                      order=(2, 1, 1),seasonal_order=(2, 1, 1, 12),
                                      steps=40)
```


    
![png](output_35_0.png)
    



```python
train_csv["date"] = pd.to_datetime(train_csv["date"])
train_csv.set_index("date", inplace=True)

test_csv["date"] = pd.to_datetime(test_csv["date"])
test_csv.set_index("date", inplace=True)
```


```python
train_csv["family"].value_counts().plot(kind='bar', figsize=(16,8))
```




    <AxesSubplot:>




    
![png](output_37_1.png)
    



```python
train_csv.iloc[:,1:].hist(figsize=(16,8))
```




    array([[<AxesSubplot:title={'center':'store_nbr'}>,
            <AxesSubplot:title={'center':'sales'}>],
           [<AxesSubplot:title={'center':'onpromotion'}>, <AxesSubplot:>]],
          dtype=object)




    
![png](output_38_1.png)
    



```python
# Create other variables
def create_date_vars(df=train_csv):
    df["year"] = df.index.year
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["day"] = df.index.day
    return df
```


```python
train_csv = create_date_vars()
test_csv = create_date_vars(df=test_csv)
train_csv.shape, test_csv.shape
```




    ((3000888, 11), (28512, 10))




```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

y_train = train_csv.pop("sales")
X_train = train_csv.iloc[:,1:]

# y_test = test_csv.pop("sales")
X_test = test_csv.iloc[:,1:]
print(X_train.shape, X_test.shape)
```

    (3000888, 9) (28512, 9)
    


```python
# Generate dummy variables for the categorical columns
def generate_dummy(data=pd.concat([X_train, X_test])):
    for col in data.select_dtypes(include='object').columns.to_list():
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)
        data.drop(col, axis=1, inplace=True) 
    return data
X = generate_dummy()
X_train = X[X.index<'2017-08-16']
X_test = X[X.index>='2017-08-16']
X_train, X_val, y_train, y_val = X_train.iloc[:-500000],X_train.iloc[-500000:],y_train.iloc[:-500000],y_train.iloc[-500000:]
```


```python
X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape
```




    ((2500888, 50), (500000, 50), (28512, 50), (2500888,), (500000,))




```python

```


```python
from sklearn.preprocessing import StandardScaler

def preprocessor(X):
    # Standardise the whole dataset
    std_scaler = StandardScaler().fit(X_train)
    D = np.copy(X)
    D = std_scaler.transform(D)
    return D
```


```python
from sklearn.preprocessing import FunctionTransformer
preprocessor_transformer = FunctionTransformer(preprocessor)
preprocessor_transformer
```




    FunctionTransformer(func=<function preprocessor at 0x0000013C38E05558>)




```python
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

p1 = Pipeline([('scaler', preprocessor_transformer),
              ('Linear Regression', XGBRegressor())])

p1
```




    Pipeline(steps=[('scaler',
                     FunctionTransformer(func=<function preprocessor at 0x0000013C38E05558>)),
                    ('Linear Regression',
                     XGBRegressor(base_score=None, booster=None, callbacks=None,
                                  colsample_bylevel=None, colsample_bynode=None,
                                  colsample_bytree=None, early_stopping_rounds=None,
                                  enable_categorical=False, eval_metric=None,
                                  gamma=None, gpu_id=None, grow_policy=None,
                                  importance_type=None,
                                  interaction_constraints=None, learning_rate=None,
                                  max_bin=None, max_cat_to_onehot=None,
                                  max_delta_step=None, max_depth=None,
                                  max_leaves=None, min_child_weight=None,
                                  missing=nan, monotone_constraints=None,
                                  n_estimators=100, n_jobs=None,
                                  num_parallel_tree=None, predictor=None,
                                  random_state=None, reg_alpha=None,
                                  reg_lambda=None, ...))])




```python
from sklearn.metrics import mean_absolute_error

def fit_and_print(p, X_train=X_train, X_test=X_val, y_train=y_train, y_test=y_val):
    # Fit the transformer
    p.fit(X_train, y_train)
    # Predict the train and test outputs
    training_prediction = p.predict(X_train)
    test_prediction =p.predict(X_test)
    
    # Print the errors
    print("Training Error:     "+str(mean_absolute_error(training_prediction, y_train)))
    print("Test Error:         "+str(mean_absolute_error(test_prediction, y_test)))
```


```python
fit_and_print(p1)
```

    Training Error:     113.11857703733529
    Test Error:         149.9649103228452
    


```python
y_pred = p1.predict(X_test)
```


```python
y_pred = pd.DataFrame({
    "y_pred": y_pred
},
index=X_test.index)
```


```python
fig, ax = plt.subplots(figsize=(15, 5))

y_train.plot(ax=ax, label="Training Set")
y_val.plot(ax=ax, label="Validation Set")
y_pred.plot(ax=ax, label="Prediction")
plt.legend()
plt.show()
```


    
![png](output_52_0.png)
    

