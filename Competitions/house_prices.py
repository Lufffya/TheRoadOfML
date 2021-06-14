import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge, LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')


''' Loading Data '''
df = pd.read_csv(r"Competitions\dataset\house_prices\train.csv")
test = pd.read_csv(r"Competitions\dataset\house_prices\test.csv")
print(df)


''' Checking for NaN value '''
# 如果 NAN 占比总行数的30% 则删除该列
null = {}
drop = []
for i in range(df.shape[1]):
    if 0 < df.iloc[:, i].isna().sum() <= df.shape[0]*(0.3):
        null[df.columns[i]] = df.iloc[:, i].isna().sum()
    elif df.iloc[:, i].isna().sum() > df.shape[0]*(0.3):
        drop.append(df.columns[i])
print(null, drop)

# 显示所有列中 NAN 的数量
nullk = list(null.keys())
nullv = [float(null[k]) for k in nullk]
plt.figure(figsize=(28, 12))
gr = sns.barplot(nullk, nullv, palette="OrRd")
gr.text(9, 200, str('Name of column \n  vs Nan values'), fontdict=dict(color='#66FF00', fontsize=40), weight='bold')
plt.show()

# 删除 NAN 列
df.drop(drop, axis=1, inplace=True)


''' Filling NaN value '''
# 显示该列的数据值分布
df.LotFrontage.plot(kind='density')
plt.show()

print(df.describe())

# 填充 NAN 使用该列的均值
df.LotFrontage.fillna(df.LotFrontage.median(), inplace=True)

print(df.MasVnrType.describe())

# 填充 NAN 使用None
df.MasVnrType.fillna('None', inplace=True)

# 填充 NAN 使用0
df.MasVnrArea.fillna(0.0, inplace=True)

# 填充 NAN 使用改列最常见的值
for i in (29, 30, 31, 32, 34):
    df.iloc[:, i].fillna(df.iloc[:, i].describe().top, inplace=True)

sns.countplot(df.Electrical, palette="OrRd")
plt.show()

# 填充 NAN 使用改列最常见的值
df.Electrical.fillna('SBrkr', inplace=True)

_, ax = plt.subplots(1, 4, figsize=(30, 10))
sns.countplot(df.GarageType, ax=ax[0], palette="OrRd")
sns.countplot(df.GarageFinish, ax=ax[1], palette="OrRd")
sns.countplot(df.GarageQual, ax=ax[2], palette="OrRd")
sns.countplot(df.GarageCond, ax=ax[3], palette="OrRd")
plt.show()

print(df.columns.get_loc('GarageType'))

for i in (56, 57, 58, 61, 62):
    print(df.iloc[:, i].dtypes)

print(df.iloc[:, 56].isna().sum())

# 填充 NAN 使用 Attchd
df.GarageType.fillna('Attchd', inplace=True)

# 填充 NAN 使用 GarageType 分组后 GarageYrBlt 的中位数
df.GarageYrBlt.fillna(df.groupby('GarageType')['GarageYrBlt'].transform('median'), inplace=True)
df.GarageFinish.fillna('UnF', inplace=True)
df.GarageQual.fillna('TA', inplace=True)
df.GarageCond.fillna('TA', inplace=True)

# 检查是否还有任何 NAN 值
null = {}
for i in range(df.shape[1]):
    if 0 < df.iloc[:, i].isna().sum():
        null[df.columns[i]] = df.iloc[:, i].isna().sum()
print(null)


''' Dummies for categorical columns '''
df = pd.get_dummies(df)
print(df)


''' Test Data '''
null = []
drop = []
for i in range(0, test.shape[1]):
    if 0 < test.iloc[:, i].isna().sum() <= test.shape[0]*(0.3):
        null.append(test.columns[i])
    elif test.iloc[:, i].isna().sum() > test.shape[0]*(0.3):
        drop.append(test.columns[i])
print(drop)

test.drop(drop, axis=1, inplace=True)

test.LotFrontage.plot(kind='density')
plt.show()

test.LotFrontage.fillna(test.LotFrontage.median(), inplace=True)
test.MasVnrType.fillna('None', inplace=True)
test.MasVnrArea.fillna(0.0, inplace=True)

for i in range(len(null)):
    if test[null[i]].dtype == object:
        test[null[i]].fillna(test[null[i]].describe().top, inplace=True)
    else:
        test[null[i]].fillna(df.groupby('MSSubClass')[null[i]].transform('median'), inplace=True)

test = pd.get_dummies(test)
print(test)


a = []
for i in list(test):
    if i not in list(df):
        a.append(i)

iD = test.Id

test.drop(a, axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)
print(test)


b = []
for i in list(df):
    if i not in list(test):
        b.append(i)

x = df.drop(b, axis=1)

y = (df.SalePrice)


_, ax = plt.subplots(1, 2, figsize=(20, 5))
a = sns.kdeplot((y), ax=ax[0])
b = sns.kdeplot(np.log(y), ax=ax[1])
b.text(13, 0.9, str('With log'), fontdict=dict(color='#02cdfb', fontsize=20))
plt.show()

y = np.log(df.SalePrice)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


rmse_score = {}

las = Lasso(alpha=0.001)
las.fit(x_train, y_train)
rmse = np.sqrt(mean_squared_error((y_test), (las.predict(x_test))))
rmse_score['Lasso'] = rmse
print(rmse)

rid = Ridge(alpha=50)
rid.fit(x_train, y_train)
rmse = np.sqrt(mean_squared_error((y_test), (rid.predict(x_test))))
rmse_score['Ridge'] = rmse
print(rmse)

rfr = RandomForestRegressor(random_state=42)
rfr.fit(x_train, y_train)
rmse = np.sqrt(mean_squared_error((y_test), (rfr.predict(x_test))))
rmse_score['Random Forest'] = rmse
print(rmse)

xgb = XGBRegressor(learning_rate=0.1)
xgb.fit(x_train, y_train)
rmse = np.sqrt(mean_squared_error((y_test), (xgb.predict(x_test))))
rmse_score['XGBoost'] = rmse
print(rmse)

lgb = LGBMRegressor(learning_rate=0.1)
lgb.fit(x_train, y_train)
rmse = np.sqrt(mean_squared_error((y_test), (lgb.predict(x_test))))
rmse_score['LightGBM'] = rmse
print(rmse)

# lor = LogisticRegression()
# lor.fit(x_train,y_train)
# rmse = np.sqrt(mean_squared_error((y_test),(lor.predict(x_test))))
# rmse_score['LogisticRegression'] = rmse
# print(rmse)

lir = LinearRegression()
lir.fit(x_train, y_train)
rmse = np.sqrt(mean_squared_error((y_test), (lir.predict(x_test))))
rmse_score['LinearRegression'] = rmse
print(rmse)

key = list(rmse_score.keys())
vals = [float(rmse_score[k]) for k in key]


''' Score comparison of different models '''
score = sns.barplot(key, vals, palette="OrRd")
for i in range(0, len(key)):
    score.text(i, vals[i]/2, str(np.round(vals[i], 4)), fontdict=dict(color='black', fontsize=10, ha='center'), weight='bold')
plt.show()

pred = np.exp(lgb.predict(test))
pred = pd.DataFrame(pred)
sub = pd.concat([iD, pred], axis=1)
sub.columns = ['Id', 'SalePrice']
sub.to_csv('submission.csv', index=False)
