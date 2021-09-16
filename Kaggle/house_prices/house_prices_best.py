import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# 读取训练数据
df_train = pd.read_csv(r'Competitions\dataset\house_prices\train.csv')

# 显示所有特征列
print(df_train.columns)

# 显示房价的描述特征
print(df_train['SalePrice'].describe())

# 显示房价的直方图
sns.distplot(df_train['SalePrice'])
plt.show()

# 分析房价数据分布的偏度(skewness)和峰度(kurtosis）
# 偏度（skewness）也称为偏态、偏态系数，是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。
# 峰度（kurtosis）又称峰态系数。表征概率密度分布曲线在平均值处峰值高低的特征数。
# 峰度反映了峰部的尖度。样本的峰度是和正态分布相比较而言统计量，如果峰度大于三，峰的形状比较尖，比正态分布峰要陡峭。反之亦然
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

# 显示 房屋地面面积/房价 关系的散点图
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.show()

# 显示 房屋地下室面积/房价 关系的散点图
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.show()

# 显示 房屋材料和装饰质量/房价 关系的箱形图
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.show()

# 显示 房屋建筑日期/房价 关系的箱形图
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
plt.show()

# 显示各个特征之间的相关性热力图
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

# 显示房价相关性特征之间相关性热力图
k = 10  # 热力图显示变量数
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# 散点图
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size=2.5)
plt.show()

# 检查缺失的数据
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

# 处理缺失数据
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
# 检查数据丢失
print(df_train.isnull().sum().max())

# 标准化数据
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

# 对特征进行两两分析
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.show()

# 删除异常的数据点
df_train.sort_values(by='GrLivArea', ascending=False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

# 对特征进行两两分析
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.show()

# 直方图和正太概率图
sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()

# 应用对数变换
df_train['SalePrice'] = np.log(df_train['SalePrice'])

# 变换后的直方图和正太概率图
sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()

# 直方图和正太概率图
sns.distplot(df_train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
plt.show()

# 数据转换
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

# 变换后的直方图和正太概率图
sns.distplot(df_train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
plt.show()

# 直方图和正太概率图
sns.distplot(df_train['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
plt.show()

# 为新变量创建列（一个就足够了，因为它是一个二元分类特征）
# if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

# 对有地下室的房屋进行数据变换，为0则不变
df_train.loc[df_train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

# 直方图和正太概率图
sns.distplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)
plt.show()

# 散点图
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
plt.show()

# 散点图
plt.scatter(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF'] > 0]['SalePrice'])
plt.show()

# 将分类变量转换为虚拟变量
df_train = pd.get_dummies(df_train)

# 训练
x = df_train.drop("SalePrice", axis=1)
y = (df_train.SalePrice)
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
