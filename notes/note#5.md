# note#5

**Q. 트리의 깊이는 어떻게 조절하고,이것이 중요한 이유는 무엇인가요?**

A. 트리의 깊이는 max_depth 매개변수를 통해 조절됩니다. 이것은 트리의 깊이를 제한하는 데 사용됩니다.
깊이가 너무 깊으면 모델이 과적합될 수 있고, 깊이가 너무 얕으면 모델이 너무 단순해질 수 있기 때문입니다.

**Q. 트리 깊이에 따라 과적합이 발생하는 이유는 무엇인가요?**

A. 과적합은 의사결정 트리가 훈련 데이터의 노이즈와 무작위 변화를 포착할 때 발생합니다.

이로 인해 보이지 않는 새로운 데이터에 대한 일반화가 제대로 이루어지지 않게 되고, 깊이가 깊은 트리는 훈련 데이터에 완벽하게 맞을 수 있지만 잘 일반화되지 않을 수 있습니다.

---

**Q. 매개변수로 사용되는 불순도 지표 중에서 "지니 불순도"와 "엔트로피"를 어떻게 선택해야 할까요? 어떤 불순도 지표가 더 효과적일까요?**

A. 지니 불순도와 엔트로피는 모두 불순도를 측정하는 지표 중 하나입니다.
지니 불순도는 계산이 빠르고 일반적으로 잘 작동하지만, 엔트로피는 정보 이득을 최대화하는 데 더 효과적일 수 있습니다.

**Q. ”지니 불순도”와 “엔트로피”의 주요 차이점에는 무엇이 있나요?**

A. 지니 불순도는 대수 계산을 포함하지 않아 속도가 빠르고 대규모 데이터 세트에 더 적합합니다.

엔트로피는 지니 불순도보다 클래스 불균형에 더 민감할 수 있습니다. 한 클래스가 지배적인 경우, 지니 불순도를 선택하는 것이 더 좋습니다.

지니 불순도와 엔트로피는 선호도의 문제인 경우가 있으며 실제로는 두 방법 모두 비슷한 결과를 보여줍니다. 

---

**Q. CART 훈련 알고리즘에서 불순도를 낮추는 게 중요한데, 그렇게 하려면 𝐽(𝑘, 𝑡𝑘 )가 작아져야 한다고 나와 있어요. 이 값이 어떤 것을 나타내는 건가요?**

A.𝐽(𝑘, 𝑡𝑘 )는 특정 노드에서 특성 𝑘와 해당 특성의 임계값 𝑡𝑘를 선택했을 때의 불순도를 나타냅니다.
이 값이 낮을수록 해당 노드의 불순도가 낮아지고, 분할이 좀 더 순수해진다고 볼 수 있어요.

**Q. 불순도를 낮추는 게 어떤 의미이며, 왜 중요한 건가요?**

A. 불순도를 낮춘다는 것은 CART 알고리즘에서 노드의 혼잡도 또는 불순도를 줄이는 것을 말합니다. 불순도란 노드 내의 데이터 포인트들이 서로 다른 클래스 또는 값을 가질 때, 해당 노드가 얼마나 혼잡한지, 불순한지 측정합니다.

CART 알고리즘에서 불순도를 낮추는 것이 모델의 정확성을 향상시키는 주요 목표 중 하나입니다.

---

**Q. 결정 트리의 규제 매개변수를 통해 모델의 복잡성을 조정할 수 있다고 들었는데, 규제를 어떻게 조절하면 되는 건가요?**

A. 규제를 조절하기 위해 max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, max_features 등의 매개변수를 사용할 수 있어요.
max_depth를 조절하면 트리의 최대 높이를 제한하고, min_samples_split과 min_samples_leaf를 조절하면 분할을 위한 최소 샘플 수나 리프 노드의 최소 샘플 수를 설정할 수 있어요.
이렇게 규제를 통해 모델을 조절하면 더 안정적이고 일반화된 결정 트리 모델을 얻을 수 있어요.

---

**Q. 회귀 작업에서 결정 트리는 어떻게 손실을 최소화하는 건가요?**

A. 결정 트리 회귀 모델은 평균제곱오차(MSE)를 최소화하는 방향으로 학습되어요.
모델은 훈련 데이터를 MSE를 최소화하는 방향으로 분할하며, 탐욕적 알고리즘을 활용합니다.

**Q. 평균제곱오차(MSE)를 최소화하고 탐욕적 알고리즘을 사용하는 이유는 무엇인가요?**

A. MSE를 최소화하는 것은 실제 값과 모델의 예측값 간의 오차를 최소화하도록 모델을 훈련시키는 것을 의미합니다. 이는 모델이 데이터에 가장 잘 맞도록 합니다. MSE를 사용하는 또 다른 이유는 미분 가능하다는 특성 때문입니다. 이는 모델 파라미터에 대한 손실 함수의 그래디언트를 계산하고 이를 사용하여 모델을 업데이트 할 수 있는것을 의미합니다.

회귀 모델에서 MSE를 최소화하는 것은 탐욕적 알고리즘의 일부로, 각 단계에서 현재 모델의 예측과 실제 값 사이 오차를 줄이는 방향으로 모델을 업데이트 합니다.

---

**<연습문제>**
**a. make_moons(n_samples=1000, noise=0.4)를 사용해 데이터셋을 생성.**

```python
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier

X,y = make_moons ( n_samples=1000, noise=0.4, random_state=42)
```

**b. 이를 train_test_split()을 사용해 훈련 세트와 테스트 세트로 나눔.**

```python
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, train_size=0.8, 
random_state=42)
```

**c. DecisionTreeClassifier의 최적의 매개변수를 찾기 위해 교차검증과 함께 그리드 탐색을 수행. (GridSearchCV 사용, max_leaf_nodes 시도)**

```python
from sklearn.model_selection import GridSearchCV

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)

grid_search_cv.fit(X_train, y_train)
```

```python
grid_search_cv.best_estimator_
#기본적으로 GridSearchCV는 전체 훈련세트로 찾은 최적의 모델을 다시 훈련시킴
#모델의 정확도를 바로 평가할 수 있다.
```

**d. 찾은 매개변수를 사용해 전체 훈련 세트에 대해 모델을 훈련시키고 테스트 세트에서 성능을 측정.**

```python
from sklearn.metrics import accuracy_score

y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)
```