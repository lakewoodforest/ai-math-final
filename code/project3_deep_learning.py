# -*- coding: utf-8 -*-
"""
[프로젝트 3] 딥러닝 - 강의평가 점수 예측 (InstEval, 5-클래스 분류)
데이터 출처: InstEval - ETH Zurich 강의평가 데이터 (R lme4 패키지 공개 데이터)
            pydataset 패키지로 로드 (https://github.com/iamaziz/PyDataset)
            원 출처: Bates et al., lme4 (https://cran.r-project.org/package=lme4)
            73,421건의 강의평가. 학생 2,972명이 교수 1,128명의 강의에 1~5점을 매긴 기록

목표: (학생 ID, 교수 ID, 학생 연차, 수강 시점, 필수과목 여부, 학과)로부터
      그 학생이 그 강의에 줄 평점(1~5)을 예측한다.

이 문제가 어려운 이유:
 - 입력의 핵심이 '누가, 누구를' 이라는 초고차원 희소 범주형 변수(합계 4,100개 ID)
 - 같은 교수의 같은 강의도 학생마다 점수가 갈리는, 사람 평가 특유의 노이즈
 - 단순 다수결(최빈 클래스)은 약 24%: 이를 얼마나 넘어서는지가 관건

수업 이론 적용:
 1) 원-핫 인코딩의 한계와 임베딩(embedding): 원-핫 벡터 x 가중치 행렬 = 행 선택임을 이용,
    4,100차원 희소 표현을 16차원 밀집 벡터로 학습 (딥러닝의 표현 학습)
 2) 다층 퍼셉트론 + ReLU + 드롭아웃 + Adam (week 13 / Keras 보조자료)
 3) 소프트맥스 + 크로스 엔트로피 (프로젝트 2와 동일한 출력층 이론)
 4) 깊은 망에서 ReLU vs 시그모이드 비교 (기울기 소실 확인 실험)
구현: Keras (TensorFlow 백엔드)
"""

import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_absolute_error

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # TF 정보성 로그 숨김
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RNG_SEED = 42
np.random.seed(RNG_SEED)
tf.random.set_seed(RNG_SEED)
BASE = pathlib.Path(__file__).resolve().parent.parent
FIG_DIR, RES_DIR = str(BASE / "figures"), str(BASE / "results")

# ---------------------------------------------------------------
# 1. 데이터 로드 및 탐색
# ---------------------------------------------------------------
df = data("InstEval")
print("데이터 크기:", df.shape)            # (73421, 7)
print(df.head())
print("평점 분포:", df["y"].value_counts().sort_index().to_dict())
# 1점 10,186 / 2점 12,951 / 3점 17,609 / 4점 16,921 / 5점 15,754 -> 최빈 클래스 비율 약 24%

n_student = df["s"].nunique()              # 2,972
n_prof    = df["d"].nunique()              # 1,128
print(f"학생 수 {n_student}, 교수 수 {n_prof}")

# 평점 분포 시각화
plt.figure(figsize=(6, 4))
cnt = df["y"].value_counts().sort_index()
plt.bar(cnt.index, cnt.values, color="steelblue")
plt.xlabel("rating (1-5)"); plt.ylabel("count")
plt.title(f"InstEval rating distribution (N={len(df):,})")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/p3_rating_dist.png", dpi=140)
plt.close()

# ---------------------------------------------------------------
# 2. 전처리
# ---------------------------------------------------------------
# 레이블: 1~5점 -> 0~4 (sparse CE는 0부터 시작하는 정수 레이블을 받음)
y = df["y"].to_numpy() - 1

# 학생/교수 ID는 값에 빈 번호가 있으므로 0..(고유수-1)의 연속 정수로 다시 매김
s_codes = df["s"].astype("category").cat.codes.to_numpy()
d_codes = df["d"].astype("category").cat.codes.to_numpy()

# 나머지 범주형 4개(학생 연차, 수강 시점, 필수 여부, 학과)는 카디널리티가 작아 원-핫 인코딩
from sklearn.preprocessing import OneHotEncoder
other = df[["studage", "lectage", "service", "dept"]].to_numpy()
oh = OneHotEncoder(sparse_output=False).fit_transform(other)   # (73421, 26)
print("원-핫 특성 차원:", oh.shape[1])

# 학습 64% / 검증 16% / 테스트 20% 분리 (클래스 비율 유지)
idx = np.arange(len(y))
idx_tmp, idx_te = train_test_split(idx, test_size=0.2, random_state=RNG_SEED, stratify=y)
idx_tr, idx_va = train_test_split(idx_tmp, test_size=0.2, random_state=RNG_SEED, stratify=y[idx_tmp])
print(f"train {len(idx_tr):,} / val {len(idx_va):,} / test {len(idx_te):,}")

def take(ids):
    """인덱스 배열로 (학생ID, 교수ID, 원-핫특성, 레이블) 묶음을 꺼내는 도우미"""
    return [s_codes[ids], d_codes[ids], oh[ids]], y[ids]

X_tr, y_tr = take(idx_tr)
X_va, y_va = take(idx_va)
X_te, y_te = take(idx_te)

EPOCHS, BATCH = 12, 512
EMB_DIM = 16          # 임베딩 차원: ID 하나를 16개의 실수로 표현

# ---------------------------------------------------------------
# 3. 모델 정의
# ---------------------------------------------------------------
def build_inputs():
    """공통 입력층: 학생 ID(정수), 교수 ID(정수), 원-핫 특성(26차원)"""
    in_s = layers.Input(shape=(), dtype="int32", name="student")
    in_d = layers.Input(shape=(), dtype="int32", name="prof")
    in_o = layers.Input(shape=(oh.shape[1],), name="others")
    return in_s, in_d, in_o

def build_baseline():
    """[기준 모델] ID를 아예 쓰지 않고 원-핫 특성 26개만으로 소프트맥스 회귀.
    '누가/누구를'이라는 정보가 없으면 어디까지 가능한지 하한선을 잰다."""
    in_o = layers.Input(shape=(oh.shape[1],), name="others")
    out = layers.Dense(5, activation="softmax")(in_o)
    return keras.Model(in_o, out)

def build_embedding_mlp(deep=False, act="relu"):
    """[주 모델] 학생/교수 ID를 임베딩으로 변환해 MLP에 연결.
    임베딩: 원-핫 벡터(2,972차원)에 가중치 행렬을 곱하면 결국 '행 하나를 꺼내는 것'과
    같으므로, 거대한 원-핫 입력 대신 행렬에서 해당 행을 직접 꺼내 쓰는 층이다.
    deep=True면 은닉층 6개를 쌓아 활성화 함수(act)에 따른 기울기 소실을 실험한다."""
    in_s, in_d, in_o = build_inputs()
    e_s = layers.Flatten()(layers.Embedding(n_student, EMB_DIM)(in_s))  # (16,)
    e_d = layers.Flatten()(layers.Embedding(n_prof, EMB_DIM)(in_d))    # (16,)
    z = layers.Concatenate()([e_s, e_d, in_o])                         # 16+16+26 = 58차원
    if deep:
        for _ in range(6):
            z = layers.Dense(64, activation=act)(z)
    else:
        z = layers.Dense(64, activation=act)(z)
        z = layers.Dropout(0.3)(z)         # 임베딩+MLP의 과적합 완화
    out = layers.Dense(5, activation="softmax")(z)
    return keras.Model([in_s, in_d, in_o], out)

def fit_model(model, name, epochs=EPOCHS, features_only=False):
    """공통 학습 함수: Adam + 크로스 엔트로피, 검증 곡선 기록.
    features_only=True면 ID 입력 없이 원-핫 특성만 넣는다(기준 모델용)."""
    xtr, xva, xte = (X_tr[2], X_va[2], X_te[2]) if features_only else (X_tr, X_va, X_te)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    hist = model.fit(xtr, y_tr, validation_data=(xva, y_va),
                     epochs=epochs, batch_size=BATCH, verbose=0)
    _, te_acc = model.evaluate(xte, y_te, verbose=0)
    print(f"{name}: test acc = {te_acc:.4f}")
    return hist.history, float(te_acc), model

# ---------------------------------------------------------------
# 4. 학습 및 실험
# ---------------------------------------------------------------
# 다수결 기준선: 무조건 최빈 클래스(3점)로 찍었을 때
maj = np.bincount(y_tr).argmax()
acc_majority = float(np.mean(y_te == maj))
print(f"다수결 기준선: {acc_majority:.4f}")

hist_a, acc_a, _      = fit_model(build_baseline(),       "A. 특성만(소프트맥스)", features_only=True)
hist_b, acc_b, _      = fit_model(build_embedding_mlp(),  "B. 임베딩+MLP (고정 12ep)")

# B'. 같은 구조 + 조기 종료(early stopping):
# B의 학습 곡선을 보면 검증 손실이 2~3 epoch에서 최저점을 찍고 다시 나빠진다(과적합).
# 검증 손실이 3 epoch 연속 개선되지 않으면 멈추고, 가장 좋았던 시점의 가중치로 되돌린다.
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
best = build_embedding_mlp()
best.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist_b2_obj = best.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                       epochs=30, batch_size=BATCH, verbose=0, callbacks=[es])
hist_b2 = hist_b2_obj.history
_, acc_b2 = best.evaluate(X_te, y_te, verbose=0)
acc_b2 = float(acc_b2)
print(f"B'. 임베딩+MLP+조기종료: test acc = {acc_b2:.4f} (멈춘 epoch: {len(hist_b2['loss'])})")

hist_c, acc_c, _      = fit_model(build_embedding_mlp(deep=True, act="relu"),    "C. 깊은 망(ReLU)", epochs=10)
hist_d, acc_d, _      = fit_model(build_embedding_mlp(deep=True, act="sigmoid"), "D. 깊은 망(sigmoid)", epochs=10)

# ---------------------------------------------------------------
# 5. 평가: 정확도 + 순서형 지표 (MAE, 인접 허용 정확도)
#    평점은 순서가 있는 레이블이므로 '몇 점 차이로 틀렸나'도 함께 본다
# ---------------------------------------------------------------
proba = best.predict(X_te, verbose=0)
pred = proba.argmax(axis=1)
mae = float(mean_absolute_error(y_te, pred))                  # 평균 몇 점 차이인가
acc_within1 = float(np.mean(np.abs(y_te - pred) <= 1))        # ±1점까지 맞다고 보면
exp_pred = (proba * np.arange(5)).sum(axis=1)                 # 확률 가중 기대 평점 (연속값)
mae_exp = float(mean_absolute_error(y_te, exp_pred))
print(f"MAE {mae:.3f}, 기대평점 MAE {mae_exp:.3f}, ±1 정확도 {acc_within1:.4f}")

cm = confusion_matrix(y_te, pred)

# ---------------------------------------------------------------
# 6. 시각화
# ---------------------------------------------------------------
# (1) 모델별 검증 정확도 곡선
plt.figure(figsize=(7.5, 4.5))
plt.axhline(acc_majority, color="gray", ls=":", label=f"majority ({acc_majority:.3f})")
plt.plot(hist_a["val_accuracy"], label="A. features-only softmax")
plt.plot(hist_b["val_accuracy"], label="B. embedding + MLP")
plt.plot(hist_c["val_accuracy"], label="C. deep ReLU")
plt.plot(hist_d["val_accuracy"], label="D. deep sigmoid")
plt.xlabel("epoch"); plt.ylabel("validation accuracy")
plt.title("Validation accuracy by model")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/p3_model_compare.png", dpi=140)
plt.close()

# (2) 주 모델의 학습/검증 곡선 (과적합 관찰)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(hist_b["loss"], label="train loss")
axes[0].plot(hist_b["val_loss"], label="val loss")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy")
axes[0].set_title("B. embedding+MLP loss"); axes[0].legend()
axes[1].plot(hist_b["accuracy"], label="train acc")
axes[1].plot(hist_b["val_accuracy"], label="val acc")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
axes[1].set_title("B. embedding+MLP accuracy"); axes[1].legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/p3_main_curve.png", dpi=140)
plt.close()

# (3) 혼동행렬: 틀려도 이웃 점수로 틀리는지 확인
plt.figure(figsize=(5.5, 5))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
for i in range(5):
    for j in range(5):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                 color="white" if cm[i, j] > cm.max()/2 else "black")
plt.xticks(range(5), [1,2,3,4,5]); plt.yticks(range(5), [1,2,3,4,5])
plt.xlabel("predicted rating"); plt.ylabel("actual rating")
plt.title("Confusion matrix (B' embedding+MLP, early stopping)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/p3_confusion.png", dpi=140)
plt.close()

# ---------------------------------------------------------------
# 7. 결과 저장
# ---------------------------------------------------------------
results = {
    "n_samples": int(len(y)),
    "n_student": int(n_student), "n_prof": int(n_prof),
    "split": {"train": len(idx_tr), "val": len(idx_va), "test": len(idx_te)},
    "rating_counts": df["y"].value_counts().sort_index().tolist(),
    "test_acc": {
        "majority": acc_majority,
        "A_features_only": acc_a,
        "B_embedding_mlp_12ep": acc_b,
        "B2_early_stopping": acc_b2,
        "C_deep_relu": acc_c,
        "D_deep_sigmoid": acc_d,
    },
    "B2_stopped_epoch": len(hist_b2["loss"]),
    "B_final": {"train_acc": hist_b["accuracy"][-1], "val_acc": hist_b["val_accuracy"][-1],
                "train_loss": hist_b["loss"][-1], "val_loss": hist_b["val_loss"][-1]},
    "mae": mae, "mae_expected": mae_exp, "acc_within1": acc_within1,
    "confusion_matrix": cm.tolist(),
}
print(json.dumps(results, indent=2))
with open(f"{RES_DIR}/p3_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("프로젝트 3 완료")
