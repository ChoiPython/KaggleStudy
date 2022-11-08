'''

생존자 예측


'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=2.5) # 이 두줄은 본 필자가 항상 쓰는 방법입니다. matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고, 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.
import missingno as msno

#ignore warnings
import warnings
warnings.filterwarnings('ignore')



df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
# print(df_train.head())

# for col in df_train.columns:
#     msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
#     print(msg)


# msno.matrix(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
# plt.show()


# f, ax = plt.subplots(1, 2, figsize=(18, 8))

# df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
# ax[0].set_title('Pie plot - Survived')
# ax[0].set_ylabel('')
# sns.countplot(x='Survived', data=df_train, ax=ax[1])
# ax[1].set_title('Count plot - Survived')

# print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count())
# print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum())

# print(pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True))

# df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()

y_position = 1.02
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])
ax[0].set_title('Number of Passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')
sns.countplot(x='Pclass', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y=y_position)

plt.show()
