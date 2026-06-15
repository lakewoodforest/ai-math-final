# -*- coding: utf-8 -*-
"""
[프로젝트 2] 다중 클래스 로지스틱 회귀 (소프트맥스 회귀) - Wine 데이터셋
데이터 출처: UCI Wine dataset (scikit-learn 내장 load_wine 으로 로드)
            https://archive.ics.uci.edu/ml/datasets/wine
            이탈리아 같은 지역에서 재배된 3개 품종(cultivar)의 와인 178개,
            화학 성분 분석 13개 특성 (알코올, 말산, 플라보노이드, 색 강도 등)

목표: 13개 화학 성분 측정값으로 와인 품종(3 클래스)을 분류

수업 이론 적용:
 1) 소프트맥스 함수로 선형 점수 -> 클래스 확률 변환
 2) 원-핫 인코딩과 크로스 엔트로피 손실
 3) 손실의 기울기 ∂L/∂W = X^T (P - Y) / N 를 직접 유도한 식으로 구현
 4) 배치 경사하강법 학습, 학습률/표준화 영향 실험
 5) scikit-learn LogisticRegression(다중 클래스)과 비교 검증
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RNG_SEED = 42
np.random.seed(RNG_SEED)
FIG_DIR, RES_DIR = "figures", "results"

# ---------------------------------------------------------------
# 1. 데이터 로드 및 탐색
# ---------------------------------------------------------------
wine = load_wine()
X, y = wine.data, wine.target               # X: (178, 13), y: 0/1/2
print("특성 이름:", wine.feature_names)
print("클래스별 개수:", np.bincount(y))      # [59, 71, 48] -> 약간의 불균형
print("특성별 평균:", np.round(X.mean(axis=0), 2))
print("특성별 표준편차:", np.round(X.std(axis=0), 2))
# proline의 표준편차(약 314)와 nonflavanoid_phenols(약 0.12)가 2600배 이상 차이
# -> 표준화 없이는 경사하강법이 제대로 동작하기 어려움을 예상할 수 있다

# 클래스가 고루 섞이도록 stratify 적용하여 8:2 분리
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=RNG_SEED, stratify=y)

# 표준화 (학습셋 통계량만 사용)
mu, sigma = X_tr.mean(axis=0), X_tr.std(axis=0)
X_tr_s = (X_tr - mu) / sigma
X_te_s = (X_te - mu) / sigma

K = 3                                        # 클래스 수
N, D = X_tr_s.shape

# ---------------------------------------------------------------
# 2. 소프트맥스 회귀 직접 구현
# ---------------------------------------------------------------
def one_hot(y, K):
    """정답 레이블을 원-핫 벡터로 변환. 예: 1 -> [0,1,0]"""
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1.0
    return Y

def softmax(U):
    """행 단위 소프트맥스. 행별 최댓값을 빼서 exp 오버플로 방지(수치 안정화)"""
    U = U - U.max(axis=1, keepdims=True)
    E = np.exp(U)
    return E / E.sum(axis=1, keepdims=True)

def cross_entropy(P, Y):
    """평균 크로스 엔트로피. log(0) 방지를 위해 아주 작은 값을 더함"""
    return float(-np.mean(np.sum(Y * np.log(P + 1e-12), axis=1)))

def train_softmax(X, y, K, lr=0.1, epochs=500):
    """배치 경사하강법으로 소프트맥스 회귀 학습.
    U = XW + b, P = softmax(U), L = CE(P, Y)
    ∂L/∂W = X^T (P - Y) / N   <- 수업에서 chain rule로 유도한 식
    ∂L/∂b = 평균(P - Y)
    """
    N, D = X.shape
    Y = one_hot(y, K)
    W = np.zeros((D, K))                     # 가중치 행렬 (13 x 3)
    b = np.zeros(K)                          # 편향 벡터
    loss_hist, acc_hist = [], []
    for ep in range(epochs):
        P = softmax(X @ W + b)               # (N, 3) 각 행이 클래스 확률
        loss_hist.append(cross_entropy(P, Y))
        acc_hist.append(float(np.mean(P.argmax(axis=1) == y)))
        G = (P - Y) / N                      # 공통 항
        W -= lr * (X.T @ G)                  # 가중치 갱신
        b -= lr * G.sum(axis=0)              # 편향 갱신
    return W, b, loss_hist, acc_hist

W, b, loss_hist, acc_hist = train_softmax(X_tr_s, y_tr, K, lr=0.5, epochs=500)

# 테스트 정확도
P_te = softmax(X_te_s @ W + b)
pred_te = P_te.argmax(axis=1)
acc_scratch = accuracy_score(y_te, pred_te)
print(f"직접 구현 테스트 정확도: {acc_scratch:.4f}")
print("혼동행렬:\n", confusion_matrix(y_te, pred_te))
print(classification_report(y_te, pred_te, target_names=wine.target_names))

# ---------------------------------------------------------------
# 3. 실험 1: 표준화를 하지 않으면?
# ---------------------------------------------------------------
W_r, b_r, loss_raw, acc_raw = train_softmax(X_tr, y_tr, K, lr=0.5, epochs=500)
pred_raw = softmax(X_te @ W_r + b_r).argmax(axis=1)
acc_no_std = accuracy_score(y_te, pred_raw)
print(f"표준화 없이 학습한 테스트 정확도: {acc_no_std:.4f}")

# 작은 학습률로 다시 시도 (발산은 피하지만 수렴이 매우 느림을 확인)
W_r2, b_r2, loss_raw2, _ = train_softmax(X_tr, y_tr, K, lr=1e-6, epochs=500)
pred_raw2 = softmax(X_te @ W_r2 + b_r2).argmax(axis=1)
acc_no_std_small_lr = accuracy_score(y_te, pred_raw2)
print(f"표준화 없음 + 작은 학습률 정확도: {acc_no_std_small_lr:.4f}")

# ---------------------------------------------------------------
# 4. 실험 2: 학습률에 따른 수렴 속도
# ---------------------------------------------------------------
plt.figure(figsize=(7, 4.5))
for lr in [0.01, 0.1, 0.5, 2.0]:
    _, _, h, _ = train_softmax(X_tr_s, y_tr, K, lr=lr, epochs=500)
    plt.plot(h, label=f"lr={lr}")
plt.xlabel("epoch"); plt.ylabel("train cross-entropy")
plt.title("Convergence by learning rate (standardized)")
plt.legend(); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/p2_lr_compare.png", dpi=140)
plt.close()

# 학습 곡선 (손실 + 정확도)
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(loss_hist, color="tab:red", label="train CE loss")
ax1.set_xlabel("epoch"); ax1.set_ylabel("cross-entropy", color="tab:red")
ax2 = ax1.twinx()
ax2.plot(acc_hist, color="tab:blue", label="train accuracy")
ax2.set_ylabel("accuracy", color="tab:blue")
plt.title("Softmax regression training curve")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/p2_training_curve.png", dpi=140)
plt.close()

# ---------------------------------------------------------------
# 5. scikit-learn 다중 클래스 로지스틱 회귀와 비교
# ---------------------------------------------------------------
sk = LogisticRegression(max_iter=5000).fit(X_tr_s, y_tr)
acc_sk = accuracy_score(y_te, sk.predict(X_te_s))
print(f"sklearn 테스트 정확도: {acc_sk:.4f}")

# ---------------------------------------------------------------
# 6. 결정경계 시각화 (특성 2개만 사용한 보조 모델)
#    13차원 전체는 그릴 수 없으므로, 판별력이 좋은 2개 특성만으로
#    수업 자료(2차원 평면의 결정경계)와 같은 방식의 그림을 그린다
# ---------------------------------------------------------------
f1, f2 = 0, 6        # alcohol, flavanoids
X2_tr = X_tr_s[:, [f1, f2]]
X2_te = X_te_s[:, [f1, f2]]
W2, b2, _, _ = train_softmax(X2_tr, y_tr, K, lr=0.5, epochs=1000)
acc_2feat = accuracy_score(y_te, softmax(X2_te @ W2 + b2).argmax(axis=1))

xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = softmax(grid @ W2 + b2).argmax(axis=1).reshape(xx.shape)

plt.figure(figsize=(7, 5.5))
plt.contourf(xx, yy, Z, alpha=0.25, levels=2, cmap="viridis")
for k, name in enumerate(wine.target_names):
    m = y_tr == k
    plt.scatter(X2_tr[m, 0], X2_tr[m, 1], s=18, label=name)
plt.xlabel("alcohol (standardized)"); plt.ylabel("flavanoids (standardized)")
plt.title(f"Decision boundary with 2 features (test acc={acc_2feat:.3f})")
plt.legend(); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/p2_decision_boundary.png", dpi=140)
plt.close()

# 혼동행렬 그림
cm = confusion_matrix(y_te, pred_te)
plt.figure(figsize=(5, 4.5))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
for i in range(K):
    for j in range(K):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black")
plt.xticks(range(K), wine.target_names); plt.yticks(range(K), wine.target_names)
plt.xlabel("predicted"); plt.ylabel("actual")
plt.title("Confusion matrix (scratch model)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/p2_confusion.png", dpi=140)
plt.close()

# ---------------------------------------------------------------
# 7. 결과 저장
# ---------------------------------------------------------------
results = {
    "n_samples": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "class_counts": np.bincount(y).tolist(),
    "acc_scratch": float(acc_scratch),
    "acc_sklearn": float(acc_sk),
    "acc_no_standardization": float(acc_no_std),
    "acc_no_std_small_lr": float(acc_no_std_small_lr),
    "acc_2features": float(acc_2feat),
    "final_train_loss": loss_hist[-1],
    "final_train_acc": acc_hist[-1],
    "confusion_matrix": cm.tolist(),
}
print(json.dumps(results, indent=2))
with open(f"{RES_DIR}/p2_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("프로젝트 2 완료")
