# Linear_regression_project

## 선형 회귀 분석으로 예측하는 영화 매출 🎞

1️⃣ 본 프로젝트는 회귀분석을 통해서 영화데이터의 매출액을 예측해보는 것이 주 목적이며, 그 안에서 어떤영화가 투자대비 인기가 있는지 등의 색다른 인 사이트를 찾아보는 것이 최종목적입니다! 👯‍♀️

2️⃣ 해당 프로젝트는 메인 데이터셋에 더하여 영화 정보를 활용한 인사이트를 찾기위해서 데이터의 전처리와 모델링을 통한 검증이 목표입니다. 

3️⃣ 데이터를 들여다보는 것에 중점을 둘 것이며, 회귀모델의 특징과 성능을 바탕으로 최적의 모델링을 구축해보았습니다. 🤜

# 💡 Project Preparation

### 🎬 Requirements

- python 3.8.5

### 🎬 Installation

```markdown
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold
```

### 🎬 Dataset

- IMDB : Internet Movie Database ([link in bio](https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset](https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset))
- Rotten Tomatoes Data([link in bio](https://www.kaggle.com/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset?select=rotten_tomatoes_movies.csv))
- GoodBooks Data([link in bio](https://www.kaggle.com/zygmunt/goodbooks-10k?select=books.csv))
- Academy dataset([link in bio](https://www.kaggle.com/unanimad/the-oscar-award))

# 💡 Project Start !

### 1. 데이터 수집 및 각 데이터 별 EDA 진행

- EDA를 통해서 인사이트를 찾기위해 노력하였습니다. 🧐

![Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled.png](Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled.png)

### 2. 데이터 가공 및 전처리

데이터 : 📂 origin_datas, 📂datas

전처리 파일 : 🗂 data-merging.ipynb, 🗂 Make_Dataset.ipynb

- 결측치 처리 및 데이터 병합

![Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%201.png](Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%201.png)

- 범주형 데이터 one_hot encoder

![Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%202.png](Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%202.png)

### 3. Regression

🎞 **전체 피쳐를 사용한 regression**

- train_test_split

![Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%203.png](Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%203.png)

```python
def display_scores(scores):
    print("r2 scores : ",scores)
    print("평균 :",scores.mean())

def get_model(model):
    
    regressor = model()
    regressor.fit(X_train, y_train)

    # 모델예측
    pred_tr = regressor.predict(X_train)
    pred_test = regressor.predict(X_test)
    rmse_tr = (np.sqrt(mean_squared_error(y_train, pred_tr)))
    rmse_test = (np.sqrt(mean_squared_error(y_test, pred_test)))
    
    print(f"{model}")
    print('RMSE of Train Data : ', round(rmse_tr, 3))
    print('Train r2_score: {:.3f}'.format(round(r2_score(y_train, pred_tr),6)))
    print('RMSE of Test Data : ', round(rmse_test,3))
    print('Test r2_score: {:.3f}'.format(round(r2_score(y_test, pred_test),6)))
    print("\n")

for model in [LinearRegression, Ridge, Lasso, RandomForestRegressor]:
    models = get_model(model)
```

![Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%204.png](Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%204.png)

RMSE 값이 크고, Cond. No. 또한 높은 상태입니다.

🎞 **VIF Factor(다중공선성) 확인 & Feature importances**

![Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%205.png](Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%205.png)

```python
model = RandomForestRegressor()
model.fit(X_train, y_train)

featureImportance = model.feature_importances_
sorted_idx = np.argsort(featureImportance)
barPos = np.arange(sorted_idx.shape[0])+.5
colnames = X.columns

plt.barh(barPos, featureImportance[sorted_idx], align='center') # (x, y) # 중요도 (y에 얼마나 영향을 미치는지)
plt.yticks(barPos, colnames[sorted_idx])
plt.tight_layout()
plt.show()
```

- 피쳐들의 분포 확인

![Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%206.png](Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%206.png)

🎞  **Feature selection & Outlier remove**

```python
from statsmodels.graphics import utils

plt.figure(figsize=(14, 10))

influence = result3.get_influence()
cooks_d2, pvals = influence.cooks_distance
fox_cr = 4 / (740 - 36 - 1)
idx = np.where(cooks_d2 > fox_cr)[0]

ax = plt.subplot()
plt.scatter(y, pred)
plt.scatter(y[idx], pred[idx], s=300, c="r", alpha=0.5)
utils.annotate_axes(range(len(idx)), idx,
                    list(zip(y[idx], pred[idx])), [(-20, 15)] * len(idx), size="small", ax=ax)
plt.title("outlier")
plt.show()
```

![Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%207.png](Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%207.png)

🎞  **Final Model**

![Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%208.png](Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%208.png)

- 최종적인 모델에서 R-squared의 값의 변동은 크지 않지만, RMSE 값이 줄어들었고 Test 값의 R2 score 값이 어느정도 증가하였습니다. 추가적으로 Cond.No의 크기를 크게 줄이면서 R2 score값과 RMSE값을 유지시킴으로서 모델의 성능을 높였습니다.

🎞  **K-Fold 교차검증**

```python
def display_scores(scores):
    print("r2 scores : ",scores)
    print("평균 :", scores.mean())
    print("표준편차:", scores.std())

def get_model(model):
    
    regressor = model()
    regressor.fit(X_train, y_train)

    # 모델예측
    pred_tr = regressor.predict(X_train)
    pred_test = regressor.predict(X_test)
    rmse_tr = (np.sqrt(mean_squared_error(y_train, pred_tr)))
    rmse_test = (np.sqrt(mean_squared_error(y_test, pred_test)))
    
    print("[cross_val]")
    full_train_scores = cross_val_score(regressor, X_train, y_train, scoring = "r2", cv=KFold(n_splits=5))
    full_test_scores = cross_val_score(regressor, X_test, y_test, scoring = "r2", cv=KFold(n_splits=5))
    print("  - cv_Train_r2")
    display_scores(full_train_scores)
    print("  - cv_Test_r2")
    display_scores(full_test_scores)
    print('\n')
```

```python
for model in [LinearRegression, Ridge, Lasso, RandomForestRegressor]:
    models = get_model(model)
```

![Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%209.png](Linear_regression_project%206d51cd21e7104c23ad4f8960d4143117/Untitled%209.png)

- 교차검증의 결과 또한 모든 모델의 교차검증에서 표준편차가 0.02대로 나타남 -> 전체적인 데이터에서 일반화된 예측성능을 보인다고 할 수 있습니다.

# 💡 Project Review

### 🎞 Regression Review

- 모든 피쳐를 스케일링과 아웃라이어의 제거 없이 모델링하였을 때의 R2-squared 값이 높았으나, 모델의 [Cond.No](http://cond.No) 값이 정상화되지 않았으며, random_state의 파라미터 설정 값에 따라서 RMSE와 R2-score의 값이 Train, Test에 따라 편차가 심해졌습니다.
- 따라서 모델 스케일링 및 피쳐셀렉션 그리고 아웃라이어의 제거를 통해 Cond.No의 값이 안정화되면서 R2-squared의 값 또한 큰 변동없이 유의미한 값을 가졌다고 볼 수 있었습니다.
- 또한, 복잡한 모델(앙상블계열의 RandomForestRegressor)와 같은 모델이 가장 뛰어난 성능을 가지고 있지만, 속도가 느리고 가역성이 떨어지는 단점이 있습니다.
- 메인 목표인 선형회귀 모델에 비롯하여, 모델의 규제를 더하는 Ridge, Lasso 등의 모델을 활용하여 overfitting 문제를 해결하는 것에 주력하였습니다.
- 교차검증을 통해 Train과 Test값에서 좀 더 신뢰할 수 있는 R2 score 값을 확인하였습니다.

### 🎞 Review

- 데이터에 대한 전처리의 필요성과 도메인 지식을 얻기위해 여러 논문을 탐색해야 할 필요성을 느꼈습니다.🧐
- 영화 매출에 대한 분석을 위해서 현재 다양한 플랫폼에서 얻어지는 수익에 대해서 데이터를 가져오고, 이를 처리한다면 좀 더 신뢰도 높은 회귀분석이 가능할것 같습니다!
- 회귀분석에 더하여 영화 장르의 자연어 처리나 classification 분석을 통해 영화라는 분야에서 새로운 Insite를 발견하면 좋을 것 같습니다.

# 💡 Built with !

- 정솔
    - EDA, 데이터 전처리, 원데이터 분석, Visualization, 1차 Regression
    - 발표자료 작성
    - GitHub: [https://github.com/solyourock](https://github.com/solyourock)
- 고현실
    - 데이터 전처리, 예측변수 스케일링, 1차 & 2차 Regression, Model 검증 및 정리
    - README 작성
    - GitHub: [https://github.com/kohyunsil](https://github.com/kohyunsil)