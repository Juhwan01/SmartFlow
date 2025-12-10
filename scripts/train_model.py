"""
ML 모델 학습 스크립트 (최종 수정버전 - XGBoost 2.0+ 호환)

1. XGBoost 2.0+ 호환성 완벽 수정 (eval_metric, early_stopping_rounds 위치 변경)
2. 비즈니스 KPI (MAE, MAPE) 중심의 학습 및 평가
3. 도메인 지식 기반 Feature Engineering 적용
"""
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from loguru import logger
import json

from src.data.data_preprocessing import ManufacturingDataProcessor


class ModelTrainer:
    """모델 학습기"""

    def __init__(self):
        self.processor = ManufacturingDataProcessor()
        self.model = None
        self.metrics = {}
        self.scaler = MinMaxScaler() # 자체 스케일러 사용

    def _calculate_mape(self, y_true, y_pred):
        """MAPE 계산 (정밀도 지표)"""
        threshold = 0.1
        mask = np.abs(y_true) > threshold
        if mask.sum() == 0:
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def feature_engineering(self, df):
        """도메인 지식 기반 변수 추가"""
        df_fe = df.copy()
        
        # 1. 용접 입열량 (Heat Input) 유사 변수
        df_fe['heat_input_proxy'] = df_fe['welding_temp1'] / (df_fe['welding_temp3'] + 1e-5)

        # 2. 압력과 온도의 상호작용
        df_fe['pressure_x_temp2'] = df_fe['welding_pressure'] * df_fe['welding_temp2']
        
        # 3. 제어 변수 합계
        df_fe['total_control'] = df_fe['welding_control1'] + df_fe['welding_control2']

        # 4. 프레스 공정의 면적/부피 유사 변수
        df_fe['press_volume_proxy'] = df_fe['press_thickness'] * df_fe['press_measurement1']

        # 5. 전체 세트포인트 대비 편차/비율 특성 생성 (타깃 열 제외)
        target_col = 'welding_strength'
        setpoint_cols = [col for col in df_fe.columns if col.endswith('_setpoint')]
        for setpoint_col in setpoint_cols:
            actual_col = setpoint_col.replace('_setpoint', '')
            if actual_col == target_col:
                continue  # 타깃 유출 방지
            if actual_col in df_fe.columns:
                error_col = f"{actual_col}_error"
                ratio_col = f"{actual_col}_ratio"
                df_fe[error_col] = df_fe[actual_col] - df_fe[setpoint_col]
                df_fe[ratio_col] = np.where(
                    df_fe[setpoint_col] != 0,
                    df_fe[actual_col] / df_fe[setpoint_col],
                    1.0
                )

        # Stage1 측정치 집계 통계
        stage_error_cols = [
            col for col in df_fe.columns
            if col.startswith('stage1_measurement') and col.endswith('_error')
        ]
        if stage_error_cols:
            stage_error_df = df_fe[stage_error_cols]
            df_fe['stage1_error_mean'] = stage_error_df.mean(axis=1)
            df_fe['stage1_error_std'] = stage_error_df.std(axis=1).fillna(0)
            df_fe['stage1_error_abs_max'] = stage_error_df.abs().max(axis=1)

        stage_ratio_cols = [
            col for col in df_fe.columns
            if col.startswith('stage1_measurement') and col.endswith('_ratio')
        ]
        if stage_ratio_cols:
            stage_ratio_df = df_fe[stage_ratio_cols]
            df_fe['stage1_ratio_mean'] = stage_ratio_df.mean(axis=1)
            df_fe['stage1_ratio_std'] = stage_ratio_df.std(axis=1).fillna(0)

        # Machine별 Derived Features
        def add_diff(col_a, col_b, new_col):
            if col_a in df_fe.columns and col_b in df_fe.columns:
                df_fe[new_col] = df_fe[col_a] - df_fe[col_b]

        add_diff('machine1_zone1_temp', 'machine1_zone2_temp', 'machine1_zone_temp_diff')
        add_diff('machine2_zone1_temp', 'machine2_zone2_temp', 'machine2_zone_temp_diff')
        add_diff('machine3_zone1_temp', 'machine3_zone2_temp', 'machine3_zone_temp_diff')
        add_diff('welding_temp5', 'welding_temp1', 'welding_temp_span')
        add_diff('combiner_temp3', 'combiner_temp1', 'combiner_temp_gradient')

        def add_power_proxy(amperage_col, rpm_col, new_col):
            if amperage_col in df_fe.columns and rpm_col in df_fe.columns:
                df_fe[new_col] = df_fe[amperage_col] * df_fe[rpm_col]

        add_power_proxy('machine1_motor_amperage', 'machine1_motor_rpm', 'machine1_power_proxy')
        add_power_proxy('machine2_motor_amperage', 'machine2_motor_rpm', 'machine2_power_proxy')
        add_power_proxy('machine3_motor_amperage', 'machine3_motor_rpm', 'machine3_power_proxy')

        # Raw material 조성 평균
        for prop_idx in range(1, 5):
            cols = [
                f"machine{machine_idx}_raw_property{prop_idx}"
                for machine_idx in (1, 2, 3)
                if f"machine{machine_idx}_raw_property{prop_idx}" in df_fe.columns
            ]
            if len(cols) >= 2:
                df_fe[f"raw_property{prop_idx}_avg"] = df_fe[cols].mean(axis=1)

        # Ambient & Welding 환경 지수
        if {'ambient_temperature', 'ambient_humidity'} <= set(df_fe.columns):
            df_fe['ambient_index'] = (
                df_fe['ambient_temperature'] * 0.7 + df_fe['ambient_humidity'] * 0.3
            )

        temp_features = [col for col in ['welding_temp1', 'welding_temp2', 'welding_temp3', 'welding_temp4', 'welding_temp5'] if col in df_fe.columns]
        if temp_features:
            df_fe['welding_temp_mean'] = df_fe[temp_features].mean(axis=1)

        machine5_temps = [col for col in ['machine5_temp3', 'machine5_temp4', 'machine5_temp5', 'machine5_temp6'] if col in df_fe.columns]
        if machine5_temps:
            df_fe['machine5_temp_mean'] = df_fe[machine5_temps].mean(axis=1)

        # Temporal Features (1-step lag, rolling mean, trend)
        temporal_cols = [
            'press_thickness',
            'stage1_error_mean',
            'welding_temp1',
            'welding_temp3',
            'welding_pressure',
            'ambient_temperature'
        ]

        for col in temporal_cols:
            if col in df_fe.columns:
                lag_col = f"{col}_lag1"
                roll_col = f"{col}_roll3"
                trend_col = f"{col}_trend"
                df_fe[lag_col] = df_fe[col].shift(1)
                df_fe[roll_col] = df_fe[col].rolling(window=3, min_periods=1).mean()
                df_fe[lag_col] = df_fe[lag_col].fillna(df_fe[col])
                df_fe[roll_col] = df_fe[roll_col].fillna(df_fe[col])
                df_fe[trend_col] = df_fe[col] - df_fe[lag_col]

        # 세트포인트 컬럼 드롭 (누설 방지)
        drop_cols = [col for col in df_fe.columns if col.endswith('_setpoint')]
        df_fe = df_fe.drop(columns=drop_cols, errors='ignore')

        # 남은 결측치 처리
        df_fe = df_fe.ffill().bfill()

        return df_fe

    def train_xgboost(
        self,
        n_estimators: int = 2000,
        max_depth: int = 8,
        learning_rate: float = 0.02
    ):
        logger.info("=" * 70)
        logger.info("XGBoost 모델 학습 시작 (KPI: MAE 최소화)")
        logger.info("=" * 70)

        # 1. 데이터 로드
        df = self.processor.create_mapped_dataset()
        target_col = "welding_strength"

        # 2. 0값 데이터 필터링
        initial_len = len(df)
        df = df[df[target_col] > 0.1].copy()
        logger.info(f"데이터 필터링: {initial_len} -> {len(df)} (유효 공정 데이터만 사용)")

        # 3. Feature Engineering 적용
        df_fe = self.feature_engineering(df)
        feature_cols = [col for col in df_fe.columns if col != target_col]

        X = df_fe[feature_cols].values
        y = df_fe[target_col].values

        # 4. 스케일링
        X_scaled = self.scaler.fit_transform(X)

        # 5. Train/Validation/Test 분리 (70/15/15)
        # Step 1: Train+Val / Test 분리
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_scaled, y, test_size=0.15, random_state=42
        )

        # Step 2: Train / Validation 분리
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.176, random_state=42  # 0.176 ≈ 15/(70+15)
        )

        logger.info(f"데이터 분할: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        # 6. Train/Val/Test 데이터 저장
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Train 데이터 저장 (RAG용)
        train_df = pd.DataFrame(X_train, columns=feature_cols)
        train_df[target_col] = y_train
        train_path = data_dir / "train_set.csv"
        train_df.to_csv(train_path, index=False)
        logger.info(f"✅ Train 데이터 저장: {train_path} ({len(train_df)} samples, RAG용)")

        # Validation 데이터 저장
        val_df = pd.DataFrame(X_val, columns=feature_cols)
        val_df[target_col] = y_val
        val_path = data_dir / "val_set.csv"
        val_df.to_csv(val_path, index=False)
        logger.info(f"✅ Validation 데이터 저장: {val_path} ({len(val_df)} samples)")

        # Test 데이터 저장 (최종 평가용)
        test_df = pd.DataFrame(X_test, columns=feature_cols)
        test_df[target_col] = y_test
        test_path = data_dir / "test_set.csv"
        test_df.to_csv(test_path, index=False)
        logger.info(f"✅ Test 데이터 저장: {test_path} ({len(test_df)} samples, 최종 평가용, 절대 재학습 금지)")

        # 6-1. Sample Weighting 계산 (불량품 강조 학습)
        # ===================================================================
        # 불균형 데이터 대응: 불량 샘플에 높은 가중치 부여
        # 참고: 2024 제조 불량 감지 연구 (MDPI Sensors)
        # ===================================================================
        # Config 기반 품질 기준 사용 (업계 표준)
        from config import settings
        LSL = settings.welding_strength_lsl  # 11.50
        USL = settings.welding_strength_usl  # 13.20

        # Train 데이터에서 불량 샘플 식별
        train_defects = (y_train < LSL) | (y_train > USL)
        num_defects = train_defects.sum()
        num_normal = len(y_train) - num_defects

        # 가중치 계산: 불량 샘플에 정상/불량 비율만큼 가중치 부여
        if num_defects > 0:
            raw_weight = num_normal / num_defects
            defect_weight = min(raw_weight, 50.0)
            if defect_weight < raw_weight:
                logger.info(
                    f"  - 불량 가중치 캡 적용: 원래 {raw_weight:.1f} -> 사용 {defect_weight:.1f}"
                )
        else:
            defect_weight = 1.0

        # Sample weights 생성
        sample_weights = np.ones(len(y_train))
        sample_weights[train_defects] = defect_weight

        logger.info(f"\n불균형 데이터 대응:")
        logger.info(f"  - 정상 샘플: {num_normal}개 (가중치: 1.0)")
        logger.info(f"  - 불량 샘플: {num_defects}개 (가중치: {defect_weight:.1f})")
        logger.info(f"  - 불량률: {num_defects/len(y_train)*100:.2f}%")

        # 7. 모델 학습 (Validation 데이터로 Early Stopping)
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=1,
            max_delta_step=1,  # 극단값 학습 개선
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            eval_metric='mae',
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )

        logger.info("\n모델 학습 중 (불량 샘플 강조 학습 + Validation Early Stopping)...")

        # Validation 데이터로 early stopping (sample_weight 적용)
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,  # 불량 샘플 강조!
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # 8. 예측 및 평가 (Train, Validation만 - Test는 evaluate_final.py에서)
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)

        self.metrics = {
            "train": {
                "mae": mean_absolute_error(y_train, y_pred_train),
                "rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "mape": self._calculate_mape(y_train, y_pred_train)
            },
            "validation": {
                "mae": mean_absolute_error(y_val, y_pred_val),
                "rmse": np.sqrt(mean_squared_error(y_val, y_pred_val)),
                "mape": self._calculate_mape(y_val, y_pred_val)
            },
            "test": {
                "note": "Test 평가는 scripts/evaluate_final.py에서 단 1회만 수행",
                "test_set_path": "data/test_set.csv"
            }
        }

        # 결과 출력
        logger.info("\n" + "=" * 70)
        logger.info("학습 결과 (Validation Set)")
        logger.info("=" * 70)
        logger.info(f"✅ MAE (평균 오차): {self.metrics['validation']['mae']:.4f} (목표: < 0.2)")
        logger.info(f"✅ MAPE (오차율)  : {self.metrics['validation']['mape']:.2f}%  (목표: < 2%)")
        logger.info("=" * 70)
        logger.info("⚠️  Test 데이터 평가는 scripts/evaluate_final.py에서 단 1회만 수행합니다.")
        logger.info("=" * 70)

        # Feature Importance
        logger.info("\n[핵심 영향 변수 - Top 5]")
        feature_importance = self.model.feature_importances_
        importance_dict = dict(zip(feature_cols, feature_importance))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_importance[:5], 1):
            logger.info(f"  {i}. {feature}: {importance:.4f}")

        return self.model, self.metrics

    def save_model(self, model_path: str = "models/quality_predictor.pkl"):
        """모델 및 관련 파일 저장"""
        if self.model is None:
            return

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open("models/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

        with open("models/metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
        with open("models/variable_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(self.processor.variable_mapping, f, indent=2, ensure_ascii=False)
            
        logger.info(f"모델 저장 완료: {model_path}")

def main():
    logger.info("SmartFlow ML Model Training (Business KPI Optimized)")

    trainer = ModelTrainer()
    model, metrics = trainer.train_xgboost()
    trainer.save_model()

    print("\n" + "=" * 70)
    print("🎯 학습 성능 요약 (Validation Set)")
    print("=" * 70)

    mae_score = metrics['validation']['mae']
    mape_score = metrics['validation']['mape']

    print(f"✅ Validation MAE  : {mae_score:.4f} (목표: < 0.2)")
    print(f"✅ Validation MAPE : {mape_score:.2f}% (목표: < 2.0%)")

    if mae_score < 0.2:
        print("\n🎉 목표 달성! 현장 투입 가능한 초정밀 예측 성능을 확보했습니다.")
        print("   (평균 오차 0.2 미만으로 품질 제어 가능)")
    else:
        print(f"\n⚠️  추가 튜닝 필요 (현재 오차: {mae_score:.4f})")

    print("=" * 70)
    print("⚠️  최종 Test 평가는 'python scripts/evaluate_final.py'로 수행하세요.")
    print("=" * 70)

if __name__ == "__main__":
    main()