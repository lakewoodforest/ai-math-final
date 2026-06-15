# -*- coding: utf-8 -*-
"""
[프로젝트 1] 선형회귀 - Diamonds 데이터셋 가격 예측
데이터 출처: ggplot2 diamonds dataset (53,940개 다이아몬드의 가격과 특성)
            pydataset 패키지를 통해 로드 (https://github.com/iamaziz/PyDataset)
            원 출처: R ggplot2 패키지 (https://ggplot2.tidyverse.org/reference/diamonds.html)

목표: 다이아몬드의 물리적 특성(캐럿, 컷, 색상, 투명도, 치수)으로 가격(USD)을 예측

수업 이론 적용:
 1) MSE(평균제곱오차) 손실함수
 2) 정규방정식(Normal Equation)으로 해석적 최적해 계산: w = (X^T X)^(-1) X^T y
 3) 배치 경사하강법(Batch Gradient Descent)으로 수치적 최적해 계산
 4) 두 방법과 scikit-learn LinearRegression 결과 비교 검증
 5) 로그 변환을 통한 성능 개선 실험
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 화면 없이 그림 파일만 저장
import matplotlib.pyplot as plt
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RNG_SEED = 42          # 재현성을 위해 난수 시드 고정
np.random.seed(RNG_SEED)

FIG_DIR = "figures"
RES_DIR = "results"

# ---------------------------------------------------------------
# 1. 데이터 로드 및 탐색 (EDA)
# ---------------------------------------------------------------
df = data("diamonds")
print("데이터 크기:", df.shape)            # (53940, 10)
print(df.head())
print(df.describe())
print("결측치 개수:\n", df.isna().sum())

# x, y, z(길이/너비/깊이 mm)가 0인 행은 물리적으로 불가능한 오류 데이터이므로 제거
# 또한 y=58.9mm, z=31.8mm 같은 기록 오류 수준의 극단값도 제거
# (처음에는 0만 제거했는데, 극단값 때문에 로그 모델의 exp 복원이 폭주하는 문제를 겪고 기준을 추가함)
n_before = len(df)
df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]
df = df[(df["y"] < 20) & (df["z"] < 20)]
print(f"이상치 제거: {n_before} -> {len(df)} 행")

# 가격 분포 시각화: 오른쪽으로 길게 치우친(skewed) 분포 확인
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(df["price"], bins=80, color="steelblue", edgecolor="none")
axes[0].set_title("Price distribution (skewed)")
axes[0].set_xlabel("price (USD)"); axes[0].set_ylabel("count")
axes[1].scatter(df["carat"], df["price"], s=2, alpha=0.15, color="darkorange")
axes[1].set_title("Carat vs Price (nonlinear)")
axes[1].set_xlabel("carat"); axes[1].set_ylabel("price (USD)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/p1_eda.png", dpi=140)
plt.close()

# ---------------------------------------------------------------
# 2. 전처리
# ---------------------------------------------------------------
# cut, color, clarity는 등급에 순서가 있는 범주형 변수(ordinal)
# -> 품질이 좋을수록 큰 정수가 되도록 직접 매핑 (도메인 지식 반영)
cut_order     = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
color_order   = {c: i for i, c in enumerate(["J", "I", "H", "G", "F", "E", "D"])}   # J(나쁨)->D(좋음)
clarity_order = {c: i for i, c in enumerate(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])}

df["cut_o"]     = df["cut"].map(cut_order)
df["color_o"]   = df["color"].map(color_order)
df["clarity_o"] = df["clarity"].map(clarity_order)

feature_cols = ["carat", "cut_o", "color_o", "clarity_o", "depth", "table", "x", "y", "z"]
X = df[feature_cols].to_numpy(dtype=float)
y = df["price"].to_numpy(dtype=float)

# 학습/테스트 분리 (8:2). 테스트셋은 학습에 일절 사용하지 않음
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RNG_SEED)

# 표준화(standardization): 각 특성을 평균 0, 표준편차 1로 변환
# 경사하강법에서 특성 간 스케일 차이가 크면 손실함수 등고선이 길쭉해져 수렴이 느려지므로 필수
mu, sigma = X_tr.mean(axis=0), X_tr.std(axis=0)
X_tr_s = (X_tr - mu) / sigma
X_te_s = (X_te - mu) / sigma        # 테스트셋도 '학습셋의' 통계량으로 변환 (데이터 누수 방지)

# 편향(bias) 항을 위한 1 열 추가 -> 행렬 한 번의 곱으로 w0 + w·x 계산 가능
ones_tr = np.ones((X_tr_s.shape[0], 1))
ones_te = np.ones((X_te_s.shape[0], 1))
Xb_tr = np.hstack([ones_tr, X_tr_s])
Xb_te = np.hstack([ones_te, X_te_s])

# ---------------------------------------------------------------
# 3. 방법 1: 정규방정식 (해석적 해)
#    MSE를 w로 미분해 0으로 두면 w* = (X^T X)^(-1) X^T y
# ---------------------------------------------------------------
w_ne = np.linalg.inv(Xb_tr.T @ Xb_tr) @ Xb_tr.T @ y_tr
pred_ne_te = Xb_te @ w_ne

# ---------------------------------------------------------------
# 4. 방법 2: 배치 경사하강법 (수치적 해)
#    L(w) = (1/N) Σ (y_i - w·x_i)^2
#    ∇L = (2/N) X^T (Xw - y),  w <- w - lr * ∇L
# ---------------------------------------------------------------
def gradient_descent(Xb, y, lr=0.1, epochs=300):
    N, d = Xb.shape
    w = np.zeros(d)                       # 가중치 0으로 초기화
    history = []                          # epoch별 MSE 기록 (수렴 확인용)
    for ep in range(epochs):
        err = Xb @ w - y                  # 예측 - 정답
        grad = (2.0 / N) * (Xb.T @ err)   # MSE의 기울기 벡터
        w -= lr * grad                    # 기울기 반대 방향으로 한 걸음
        history.append(float(np.mean(err ** 2)))
    return w, history

w_gd, hist = gradient_descent(Xb_tr, y_tr, lr=0.1, epochs=300)
pred_gd_te = Xb_te @ w_gd

# 학습률 비교 실험: 너무 작으면 느리고, 너무 크면 발산함을 확인
lr_exp = {}
for lr in [0.001, 0.01, 0.1, 1.0]:
    _, h = gradient_descent(Xb_tr, y_tr, lr=lr, epochs=300)
    lr_exp[lr] = h

plt.figure(figsize=(7, 4.5))
CAP = 1e9                                    # 발산한 손실은 그림 범위를 위해 1e9까지만 표시
for lr, h in lr_exp.items():
    h = np.asarray(h)
    ok = np.isfinite(h) & (h < CAP)          # lr=1.0은 손실이 inf로 폭주하므로 범위 내 값만 그림
    n_ok = int(ok.sum())
    label = f"lr={lr}" + ("" if n_ok == len(h) else " (diverges)")
    plt.plot(np.arange(len(h))[ok], h[ok], label=label)
plt.yscale("log")
plt.xlabel("epoch"); plt.ylabel("train MSE (log scale)")
plt.title("Gradient descent convergence by learning rate")
plt.legend(); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/p1_lr_compare.png", dpi=140)
plt.close()

# ---------------------------------------------------------------
# 5. 방법 3: scikit-learn LinearRegression (검증용 기준)
# ---------------------------------------------------------------
sk = LinearRegression().fit(X_tr_s, y_tr)
pred_sk_te = sk.predict(X_te_s)

# 세 방법의 가중치가 (거의) 일치하는지 확인 -> 구현이 올바르다는 근거
print("정규방정식 w[:4]:", np.round(w_ne[:4], 3))
print("경사하강법 w[:4]:", np.round(w_gd[:4], 3))
print("sklearn   w[:4]:", np.round(np.r_[sk.intercept_, sk.coef_[:3]], 3))

# ---------------------------------------------------------------
# 6. 개선 실험: 목표값 로그 변환
#    가격 분포가 한쪽으로 치우쳐 있고 carat-price 관계가 비선형이므로
#    log(price)를 예측하면 선형 모형의 가정에 더 가까워진다
# ---------------------------------------------------------------
ylog_tr, ylog_te = np.log(y_tr), np.log(y_te)
w_log = np.linalg.inv(Xb_tr.T @ Xb_tr) @ Xb_tr.T @ ylog_tr
pred_log_space = Xb_te @ w_log           # 로그 공간에서의 예측값
pred_log_te = np.exp(pred_log_space)     # 로그 공간 예측을 다시 달러로 환원
r2_log_space = r2_score(ylog_te, pred_log_space)   # 로그 공간 자체의 설명력도 따로 기록

# ---------------------------------------------------------------
# 7. 평가 및 결과 저장
# ---------------------------------------------------------------
def metrics(y_true, y_pred):
    return {
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

results = {
    "n_samples": int(len(df)),
    "n_features": len(feature_cols),
    "normal_equation": metrics(y_te, pred_ne_te),
    "gradient_descent": metrics(y_te, pred_gd_te),
    "sklearn": metrics(y_te, pred_sk_te),
    "log_target": metrics(y_te, pred_log_te),
    "log_target_logspace_R2": float(r2_log_space),
    "weights_ne": {c: float(w) for c, w in zip(["bias"] + feature_cols, w_ne)},
    "gd_final_train_mse": hist[-1],
}
print(json.dumps(results, indent=2))
with open(f"{RES_DIR}/p1_results.json", "w") as f:
    json.dump(results, f, indent=2)

# 예측 vs 실제, 잔차 그림
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(y_te, pred_ne_te, s=2, alpha=0.15)
lim = [0, y_te.max()]
axes[0].plot(lim, lim, "r--", lw=1)
axes[0].set_xlabel("actual price"); axes[0].set_ylabel("predicted price")
axes[0].set_title(f"Linear model (R2={results['normal_equation']['R2']:.3f})")
axes[1].scatter(y_te, pred_log_te, s=2, alpha=0.15, color="green")
axes[1].plot(lim, lim, "r--", lw=1)
axes[1].set_xlabel("actual price"); axes[1].set_ylabel("predicted price")
axes[1].set_title(f"Log-target model (R2={results['log_target']['R2']:.3f})")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/p1_pred_vs_actual.png", dpi=140)
plt.close()

# 잔차(residual) 분석: 잔차가 0 주위에 고르게 퍼져 있어야 좋은 모형
plt.figure(figsize=(7, 4.5))
plt.scatter(pred_ne_te, y_te - pred_ne_te, s=2, alpha=0.15)
plt.axhline(0, color="r", ls="--", lw=1)
plt.xlabel("predicted price"); plt.ylabel("residual (actual - predicted)")
plt.title("Residual plot (heteroscedasticity visible)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/p1_residual.png", dpi=140)
plt.close()

print("프로젝트 1 완료")
