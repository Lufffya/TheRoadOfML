#
# https://www.kaggle.com/startupsci/titanic-data-science-solutions
#

''' 工作流阶段 '''
# 竞赛解决方案工作流程经历了数据科学解决方案书中描述的七个阶段
# 1 问题或问题定义
# 2 获取培训和测试数据
# 3 整理、准备、清理数据
# 4 分析、识别模式并探索数据
# 5 建模、预测和解决问题
# 6 可视化、报告并呈现问题解决步骤和最终解决方案
# 7 供或提交结果

''' 工作流目标 '''
# 数据科学解决方案工作流解决了七个主要目标
# 1 分类。 我们可能想要对我们的样本进行分类或分类。 我们可能还想了解不同类别与我们的解决方案目标的含义或相关性。
# 2 相关。 可以根据训练数据集中的可用特征来解决问题。 数据集中的哪些特征对我们的解决方案目标有重大贡献？ 
# 从统计上讲，特征和解决方案目标之间是否存在相关性？ 随着特征值的变化，解决方案的状态也会发生变化，反之亦然？ 
# 这可以针对给定数据集中的数值和分类特征进行测试。 我们可能还想为后续目标和工作流程阶段确定除生存之外的特征之间的相关性。 关联某些特征可能有助于创建、完成或修正特征。
# 3 转换。 对于建模阶段，需要准备数据。 根据模型算法的选择，可能需要将所有特征转换为数值等效值。 例如，将文本分类值转换为数值。
# 4 完成。 数据准备还可能需要我们估计特征中的任何缺失值。 当没有缺失值时，模型算法可能效果最好。
# 5 纠正。 我们还可以分析给定训练数据集的错误或特征中可能不准确的值，并尝试校正这些值或排除包含错误的样本。 
# 一种方法是检测我们的样本或特征中的任何异常值。 如果某个特征对分析没有贡献或可能显着扭曲结果，我们也可能会完全丢弃该特征。
# 6 创造。 我们是否可以基于现有特征或一组特征创建新特征，以便新特征遵循相关性、转换、完整性目标。
# 7 制图。 如何根据数据的性质和解决方案目标选择正确的可视化图和图表。



# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# 获取数据
# Python Pandas 包帮助我们处理数据集。我们首先将训练和测试数据集获取到 Pandas DataFrames 中。我们还将这些数据集结合起来对两个数据集运行某些操作。
train_df = pd.read_csv(r"Competitions\dataset\titanic\train.csv")
test_df = pd.read_csv(r"Competitions\dataset\titanic\test.csv")
combine = [train_df, test_df]

# 通过描述数据进行分析
# Pandas 还有助于描述在我们项目早期回答以下问题的数据集。
# 数据集中有哪些特征可用？
# 注意直接操作或分析这些的特征名称。这些功能名称在此处的 Kaggle 数据页面上进行了描述。
print(train_df.columns.values)

# 哪些特征是分类的？
# 这些值将样本分类为类似样本的集合。在分类特征中是基于名义、有序、比率或区间的值吗？除其他外，这有助于我们选择合适的图表进行可视化。
# 分类： Survived, Sex, and Embarked. Ordinal: Pclass.
# 哪些特征是数字的？
# 哪些特征是数字的？这些值因样本而异。在数值特征中，数值是离散的、连续的还是基于时间序列的？除其他外，这有助于我们选择合适的图表进行可视化。
# 连续：Age, Fare. Discrete: SibSp, Parch.
print(train_df.head())

# 哪些特征是混合数据类型？
# 同一特征中的数字、字母数字数据。这些是纠正目标的候选人。
# 工单是数字和字母数字数据类型的混合。 Cabin 是字母数字的。
# 哪些功能可能包含错误或拼写错误？
# 这对于大型数据集来说更难检查，但是从较小的数据集中检查一些样本可能会直接告诉我们哪些特征可能需要更正。
# 名称特征可能包含错误或拼写错误，因为用于描述名称的方法有多种，包括标题、圆括号和用于替代名称或简称的引号。
print(train_df.tail())

# 哪些特征包含空白、空值或空值？
# 这些都需要纠正。
# Cabin > Age > Embarked 特征按训练数据集的顺序包含许多空值。
# Cabin > Age 在测试数据集的情况下是不完整的。
# 各种特征的数据类型是什么？
# 在转换目标期间帮助我们。
# 七个特征是整数或浮点数。六个在测试数据集的情况下。
# 五个特征是字符串（对象）。
print(train_df.info())
print('_'*40)
print(test_df.info())

# 样本中数值特征值的分布是什么？
# 这有助于我们在其他早期见解中确定实际问题域的训练数据集的代表性。
# 样本总数为 891 人，即泰坦尼克号上实际乘客人数（2,224 人）的 40%。
# Survived 是一个具有 0 或 1 值的分类特征。
# 大约 38% 的样本存活，代表实际存活率为 32%。
# 大多数乘客 (> 75%) 没有与父母或孩子一起旅行。
# 近 30% 的乘客有兄弟姐妹和/或配偶。
# 票价差异很大，很少有乘客 (<1%) 支付高达 512 美元的费用。
# 65-80 岁的老年乘客很少（<1%）。
print(train_df.describe())

# 分类特征的分布是什么？
# 名称在整个数据集中是唯一的（count=unique=891）
# 性别变量作为两个可能的值，男性占 65%（顶部 = 男性，频率 = 577/计数 = 891）。
# Cabin 值在样本中有多个重复项。或者，几个乘客共用一个客舱。
# Embarked 取三个可能的值。大多数乘客使用的 S 端口（顶部 = S）
# 工单功能具有高比例（22%）的重复值（unique=681）。
print(train_df.describe(include=['O']))

# 基于数据分析的假设
# 我们根据迄今为止所做的数据分析得出以下假设。在采取适当行动之前，我们可能会进一步验证这些假设。
# 相关。我们想知道每个特征与生存的相关性如何。我们希望在我们的项目早期就这样做，并在项目后期将这些快速相关性与建模相关性相匹配。
# 完成。我们可能想要完成年龄特征，因为它肯定与生存相关。我们可能希望完成 Embarked 功能，因为它也可能与生存或其他重要功能相关。
# 纠正。票证功能可能会从我们的分析中删除，因为它包含很高的重复率 (22%)，并且票证与生存之间可能没有相关性。
# Cabin 特征可能会被删除，因为它非常不完整，或者在训练和测试数据集中包含许多空值。
# 乘客 ID 可能会从训练数据集中删除，因为它对生存没有贡献。名字特征比较不规范，可能对生存没有直接的帮助，所以可能会掉线。
# 创造。我们可能希望基于 Parch 和 SibSp 创建一个名为 Family 的新功能，以获取船上家庭成员的总数。
# 我们可能想要设计 Name 特征来提取 Title 作为一个新特征。
# 我们可能想为年龄带创建新功能。这将连续的数字特征转换为有序的分类特征。
# 如果有助于我们的分析，我们可能还想创建票价范围功能。
# 分类。我们还可以根据前面提到的问题描述添加到我们的假设中。
# 女性（性别=女性）更有可能幸存下来。
# 儿童（年龄<？）更有可能幸存下来。
# 上层乘客（Pclass=1）更有可能幸存下来。


# 通过旋转特征进行分析
# 为了确认我们的一些观察和假设，我们可以通过相互转换特征来快速分析我们的特征相关性。我们只能在这个阶段对没有任何空值的特征这样做。仅对分类 (Sex)、有序 (Pclass) 或离散 (SibSp, Parch) 类型的特征这样做也是有意义的。
# Pclass 我们观察到 Pclass=1 和 Survived（分类 #3）之间的显着相关性（>0.5）。我们决定在我们的模型中包含此功能。
# 性别 我们确认在问题定义过程中观察到的性别=女性的存活率非常高，为 74%（分类 #1）。
# SibSp 和 Parch 这些特征对于某些值具有零相关性。最好从这些单个特征中导出一个特征或一组特征（创建#1）。
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# 通过可视化数据进行分析
# 现在我们可以继续使用可视化分析数据来确认我们的一些假设。
# 关联数值特征
# 让我们首先了解数值特征与我们的解决方案目标（幸存）之间的相关性。
# 直方图对于分析连续的数值变量（如年龄）很有用，其中条带或范围将有助于识别有用的模式。直方图可以使用自动定义的 bin 或相等范围的波段来指示样本的分布。这有助于我们回答与特定波段相关的问题（婴儿的存活率是否更高？）
# 请注意，直方图可视化中的 x 轴表示样本或乘客的数量。
# 观察。
# 婴儿（年龄<=4）具有高存活率。
# 最年长的乘客（年龄 = 80）幸存下来。
# 大量 15-25 岁的人没有存活下来。
# 大多数乘客年龄在 15-35 岁之间。
# 决定。
# 这个简单的分析证实了我们作为后续工作流程阶段决策的假设。
# 我们应该在模型训练中考虑年龄（我们的假设分类 #2）。
# 完成空值的年龄特征（完成#1）。
# 我们应该捆绑年龄组（创建#3）。
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()

# 关联数值和序数特征
# 我们可以组合多个特征来使用单个图识别相关性。这可以通过具有数值的数值和分类特征来完成。
# 观察。
# Pclass=3 有最多的乘客，但大多数没有幸存。证实了我们的分类假设 #2。
# Pclass=2 和 Pclass=3 的婴儿乘客大部分幸存下来。进一步验证了我们的分类假设 #2。
# Pclass=1 中的大多数乘客幸免于难。证实了我们的分类假设 #3。
# Pclass 因乘客年龄分布而异。
# 决定。
# 考虑使用 Pclass 进行模型训练。

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()

# 关联分类特征
# 现在我们可以将分类特征与我们的解决方案目标相关联。
# 观察。
# 女性乘客的存活率远高于男性。确认分类（#1）。
# Embarked=C 中的例外，其中男性的存活率更高。这可能是 Pclass 和 Embarked 之间的相关性，反过来 Pclass 和 Survived 之间的相关性，不一定是 Embarked 和 Survived 之间的直接相关性。
# 对于 C 和 Q 端口，与 Pclass=2 相比，男性在 Pclass=3 时具有更好的存活率。完成（#2）。
# 对于 Pclass=3 和男性乘客，登船港的存活率各不相同。相关（#1）。
# 决定。
# 在模型训练中添加性别功能。
# 完成并将 Embarked 功能添加到模型训练中。

# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()

# 关联分类特征和数值特征
# 我们可能还想将分类特征（具有非数字值）和数字特征相关联。我们可以考虑将 Embarked（分类非数字）、Sex（分类非数字）、Fare（连续数字）与 Survived（分类数字）相关联。
# 观察。
# 更高的付费乘客有更好的生存。确认我们创建（#4）票价范围的假设。
# 登船港与存活率相关。确认相关（#1）和完成（#2）。
# 决定。
# 考虑捆绑票价功能。

# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
plt.show()

# 纠缠数据
# 我们收集了一些关于我们的数据集和解决方案要求的假设和决定。到目前为止，我们无需更改单个功能或值即可实现这些。现在让我们执行我们的决定和假设，以纠正、创建和完成目标。
# 通过删除特征进行校正
# 这是一个很好的开始执行目标。通过删除特征，我们处理的数据点更少。加速我们的笔记本并简化分析。
# 根据我们的假设和决定，我们希望放弃 Cabin（更正 #2）和 Ticket（更正 #1）功能。
# 请注意，在适用的情况下，我们同时对训练和测试数据集执行操作以保持一致。

print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

# 创建从现有特征中提取的新特征
# 我们想分析是否可以设计 Name 特征来提取标题并测试标题与生存之间的相关性，然后再删除 Name 和 PassengerId 特征。
# 在以下代码中，我们使用正则表达式提取 Title 特征。 RegEx 模式 (\w+\.) 匹配名称特征中以点字符结尾的第一个单词。 expand=False 标志返回一个 DataFrame。
# 观察。
# 当我们绘制 Title、Age 和 Survived 时，我们注意到以下观察结果。
# 大多数标题都准确地结合了年龄组。例如：硕士职称的平均年龄为 5 年。
# Title Age 乐队的生存率略有不同。
# 某些头衔大多幸存下来（Mme、Lady、Sir）或没有（Don、Rev、Jonkheer）。
# 决定。
# 我们决定为模型训练保留新的 Title 功能。

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train_df['Title'], train_df['Sex']))

# 我们可以用更常见的名称替换许多标题或将它们归类为稀有。
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# 我们可以将分类标题转换为序数
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
print(train_df.head())

# 现在我们可以安全地从训练和测试数据集中删除 Name 特征。我们也不需要训练数据集中的PassengerId 特征
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
print(train_df.shape, test_df.shape)

# 转换分类特征
# 现在我们可以将包含字符串的特征转换为数值。这是大多数模型算法所必需的。这样做也将帮助我们实现功能完成目标。
# 让我们首先将 Sex 特征转换为名为 Gender 的新特征，其中女性 = 1，男性 = 0。
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

print(train_df.head())

# 完成一个数值连续特征
# 现在我们应该开始估计和完成缺失值或空值的特征。我们将首先为年龄功能执行此操作。
# 我们可以考虑三种方法来完成一个数值连续特征。
# 一种简单的方法是在均值和标准差之间生成随机数。
# 猜测缺失值的更准确方法是使用其他相关特征。在我们的例子中，我们注意到年龄、性别和 Pclass 之间的相关性。使用跨 Pclass 和 Gender 特征组合集的 Age 中值来猜测 Age 值。因此，Pclass=1 和 Gender=0、Pclass=1 和 Gender=1 的中位数年龄，依此类推...
# 结合方法 1 和 2。因此，不要根据中位数猜测年龄值，而是根据 Pclass 和 Gender 组合使用平均值和标准差之间的随机数。
# 方法 1 和 3 将在我们的模型中引入随机噪声。多次执行的结果可能会有所不同。我们更喜欢方法2。

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()

# 让我们从准备一个空数组开始，以包含基于 Pclass x Gender 组合的猜测年龄值。
guess_ages = np.zeros((2,3))
print(guess_ages)

# 现在我们迭代 Sex (0 or 1) 和 Pclass (1, 2, 3) 来计算六种组合的 Age 猜测值。
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

print(train_df.head())

# 让我们创建年龄带并确定与 Survived 的相关性。
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

# 让我们用基于这些带的序数替换 Age 。
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
    
print(train_df.head())

# 我们无法删除 AgeBand 功能。
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
print(train_df.head())

# 创建结合现有特征的新特征
# 我们可以为 FamilySize 创建一个新功能，它结合了 Parch 和 SibSp。这将使我们能够从我们的数据集中删除 Parch 和 SibSp。
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# 我们可以创建另一个名为 IsAlone 的功能。
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

# 让我们放弃 Parch、SibSp 和 FamilySize 功能，转而使用 IsAlone。
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

print(train_df.head())

# 我们还可以创建一个结合 Pclass 和 Age 的人工特征。
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))

# 完成分类特征
# 登船特征采用基于登船港的 S、Q、C 值。我们的训练数据集有两个缺失值。我们只是用最常见的情况填充这些。
freq_port = train_df.Embarked.dropna().mode()[0]
print(freq_port)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# 将分类特征转换为数值
# 我们现在可以通过创建一个新的数字端口特征来转换 EmbarkedFill 特征
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

print(train_df.head())

# 快速完成和转换数字特征
# 我们现在可以使用模式完成测试数据集中单个缺失值的票价特征，以获得该特征最常出现的值。我们在一行代码中完成此操作。
# 请注意，我们不会创建中间新特征或对相关性进行任何进一步分析来猜测缺失的特征，因为我们仅替换了单个值。完成目标实现了模型算法对非空值进行操作的预期要求。
# 我们可能还希望将票价四舍五入到两位小数，因为它代表货币。
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
print(test_df.head())

# 我们无法创建 FareBand。
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

# 根据 FareBand 将 Fare 特征转换为序号值
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
print(train_df.head(10))

# 和测试数据集。
test_df.head(10)

# 建模、预测和求解
# 现在我们已准备好训练模型并预测所需的解决方案。
# 有 60 多种预测建模算法可供选择。我们必须了解问题的类型和解决方案要求，以缩小到我们可以评估的几个模型。
# 我们的问题是分类和回归问题。我们想确定输出（是否幸存）与其他变量或特征（性别、年龄、港口...）之间的关系。
# 我们还在执行一类机器学习，称为监督学习，因为我们正在使用给定的数据集训练我们的模型。
# 有了这两个标准——监督学习加上分类和回归，我们可以将模型的选择范围缩小到几个。这些包括：
# 逻辑回归
# KNN 或 k-最近邻
# 支持向量机
# 朴素贝叶斯分类器
# 决策树
# 随机森林
# 感知器
# 人工神经网络
# RVM 或相关向量机

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)

# 逻辑回归是在工作流早期运行的有用模型。 
# Logistic 回归通过使用逻辑函数（即累积逻辑分布）估计概率来衡量分类因变量（特征）与一个或多个自变量（特征）之间的关系。参考维基百科。
# 请注意模型根据我们的训练数据集生成的置信度分数。

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)


# 我们可以使用 Logistic 回归来验证我们对创建特征和完成目标的假设和决策。这可以通过计算决策函数中特征的系数来完成。
# 正系数会增加响应的对数优势（从而增加概率），而负系数会降低响应的对数优势（从而降低概率）。
# Sex是最高的正系数，意味着随着Sex值的增加（男性：0到女性：1），Survived=1的概率增加最多。
# 相反，随着 Pclass 的增加，Survived=1 的概率降低得最多。
# 这种方式 Age*Class 是一个很好的人工建模特征，因为它与 Survived 具有第二高的负相关性。
# 标题也是第二高的正相关。

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

print(coeff_df.sort_values(by='Correlation', ascending=False))

# 接下来，我们使用支持向量机进行建模，支持向量机是具有相关学习算法的监督学习模型，用于分析用于分类和回归分析的数据。
# 给定一组训练样本，每个样本都标记为属于两个类别中的一个或另一个，SVM 训练算法构建一个模型，将新的测试样本分配给一个或另一个类别，使其成为非概率二元线性分类器。参考维基百科。

# 请注意，该模型生成的置信度得分高于物流回归模型。

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print(acc_svc)

# 在模式识别中，k-Nearest Neighbors 算法（或简称 k-NN）是一种用于分类和回归的非参数方法。
# 一个样本通过其邻居的多数投票进行分类，样本被分配到其 k 个最近邻居中最常见的类（k 是一个正整数，通常很小）。
# 如果 k = 1，则对象被简单地分配给该单个最近邻居的类。参考维基百科。

# KNN 置信度得分比物流回归好，但比 SVM 差。

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(acc_knn)


# 在机器学习中，朴素贝叶斯分类器是一系列简单的概率分类器，它基于应用贝叶斯定理和特征之间的强（朴素）独立假设。
# 朴素贝叶斯分类器具有高度可扩展性，在学习问题中需要许多与变量（特征）数量成线性关系的参数。参考维基百科。

# 模型生成的置信度得分是迄今为止评估的模型中最低的。

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print(acc_gaussian)

# 感知器是一种对二元分类器（可以决定由数字向量表示的输入是否属于某个特定类的函数）进行监督学习的算法。
# 它是一种线性分类器，即基于将一组权重与特征向量相结合的线性预测函数进行预测的分类算法。该算法允许在线学习，因为它一次处理一个训练集中的元素。参考维基百科。

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print(acc_perceptron)


# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(acc_linear_svc)


# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print(acc_sgd)

# 该模型使用决策树作为预测模型，将特征（树枝）映射到关于目标值（树叶）的结论。
# 目标变量可以取一组有限值的树模型称为分类树；在这些树结构中，叶子代表类标签，分支代表导致这些类标签的特征的连接。
# 目标变量可以采用连续值（通常是实数）的决策树称为回归树。参考维基百科。

# 模型置信度得分是迄今为止评估的模型中最高的。

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(acc_decision_tree)


# 下一个模型随机森林是最受欢​​迎的模型之一。
# 随机森林或随机决策森林是一种用于分类、回归和其他任务的集成学习方法，它通过在训练时构建大量决策树（n_estimators=100）并输出作为类的众数的类（分类）来进行操作或单个树的平均预测（回归）。参考维基百科。

# 模型置信度得分是迄今为止评估的模型中最高的。我们决定使用此模型的输出 (Y_pred) 来创建我们的竞赛提交结果。

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)

# 模型评估
# 我们现在可以对所有模型的评估进行排名，以选择最适合我们问题的模型。虽然决策树和随机森林的得分相同，但我们选择使用随机森林，因为它们可以纠正决策树过度拟合训练集的习惯。

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})

print(models.sort_values(by='Score', ascending=False))

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)

# 我们提交给竞赛网站 Kaggle 的结果是在 6,082 份参赛作品中获得了 3,883 分。
# 该结果在比赛进行时具有指示性。这个结果只占提交数据集的一部分。对于我们的第一次尝试来说还不错。任何提高我们分数的建议都是最受欢迎的。

# 参考
# 该笔记本是根据解决泰坦尼克号竞赛和其他来源的出色工作而创建的。