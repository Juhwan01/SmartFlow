"""
실제 Multi-Stage 제조 데이터 전처리 및 변수 매핑

Kaggle: Multi-Stage Continuous-Flow Manufacturing Process
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from loguru import logger
import pickle


class ManufacturingDataProcessor:
    """제조 데이터 전처리기"""

    def __init__(self, csv_path: str = "src/data/continuous_factory_process.csv"):
        """
        Args:
            csv_path: CSV 파일 경로
        """
        self.csv_path = csv_path
        self.data = None
        self.scaler = MinMaxScaler()

        # 변수 매핑 정의 (기획서 기반)
        self.variable_mapping = {
            # Stage 1 Output (프레스 공정 출력) - 주요 측정값
            "press_thickness": "Stage1.Output.Measurement0.U.Actual",  # 두께 (주요 관심 변수)
            "press_thickness_setpoint": "Stage1.Output.Measurement0.U.Setpoint",
            "press_measurement1": "Stage1.Output.Measurement1.U.Actual",
            "press_measurement1_setpoint": "Stage1.Output.Measurement1.U.Setpoint",
            "press_measurement2": "Stage1.Output.Measurement2.U.Actual",
            "press_measurement2_setpoint": "Stage1.Output.Measurement2.U.Setpoint",

            # Machine4 (용접 공정 제어 변수)
            "welding_temp1": "Machine4.Temperature1.C.Actual",  # 전류에 해당 (온도와 상관)
            "welding_temp2": "Machine4.Temperature2.C.Actual",
            "welding_pressure": "Machine4.Pressure.C.Actual",  # 압력
            "welding_temp3": "Machine4.Temperature3.C.Actual",  # 속도에 해당

            # Machine5 (용접 공정 추가 제어)
            "welding_control1": "Machine5.Temperature1.C.Actual",
            "welding_control2": "Machine5.Temperature2.C.Actual",

            # Stage 2 Output (용접 공정 출력) - 예측 타겟만 사용
            "welding_strength": "Stage2.Output.Measurement0.U.Actual",  # 용접 강도 (주요 타겟)
            "welding_strength_setpoint": "Stage2.Output.Measurement0.U.Setpoint",
            # 주의: welding_measurement1, welding_measurement2는 Data Leakage를 유발하므로 제거!
        }

        additional_mapping = {
            # 공정 환경
            "ambient_temperature": "AmbientConditions.AmbientTemperature.U.Actual",
            "ambient_humidity": "AmbientConditions.AmbientHumidity.U.Actual",

            # Combiner 단계
            "combiner_temp1": "FirstStage.CombinerOperation.Temperature1.U.Actual",
            "combiner_temp2": "FirstStage.CombinerOperation.Temperature2.U.Actual",
            "combiner_temp3": "FirstStage.CombinerOperation.Temperature3.C.Actual",

            # Machine4 확장 센서
            "welding_temp4": "Machine4.Temperature4.C.Actual",
            "welding_temp5": "Machine4.Temperature5.C.Actual",
            "welding_exit_temp": "Machine4.ExitTemperature.U.Actual",

            # Machine5 확장 센서
            "machine5_temp3": "Machine5.Temperature3.C.Actual",
            "machine5_temp4": "Machine5.Temperature4.C.Actual",
            "machine5_temp5": "Machine5.Temperature5.C.Actual",
            "machine5_temp6": "Machine5.Temperature6.C.Actual",
            "machine5_exit_temp": "Machine5.ExitTemperature.U.Actual",

            # Machine1 공정 변수
            "machine1_raw_property1": "Machine1.RawMaterial.Property1",
            "machine1_raw_property2": "Machine1.RawMaterial.Property2",
            "machine1_raw_property3": "Machine1.RawMaterial.Property3",
            "machine1_raw_property4": "Machine1.RawMaterial.Property4",
            "machine1_feeder_param": "Machine1.RawMaterialFeederParameter.U.Actual",
            "machine1_zone1_temp": "Machine1.Zone1Temperature.C.Actual",
            "machine1_zone2_temp": "Machine1.Zone2Temperature.C.Actual",
            "machine1_motor_amperage": "Machine1.MotorAmperage.U.Actual",
            "machine1_motor_rpm": "Machine1.MotorRPM.C.Actual",
            "machine1_material_pressure": "Machine1.MaterialPressure.U.Actual",
            "machine1_material_temp": "Machine1.MaterialTemperature.U.Actual",
            "machine1_exit_temp": "Machine1.ExitZoneTemperature.C.Actual",

            # Machine2 공정 변수
            "machine2_raw_property1": "Machine2.RawMaterial.Property1",
            "machine2_raw_property2": "Machine2.RawMaterial.Property2",
            "machine2_raw_property3": "Machine2.RawMaterial.Property3",
            "machine2_raw_property4": "Machine2.RawMaterial.Property4",
            "machine2_feeder_param": "Machine2.RawMaterialFeederParameter.U.Actual",
            "machine2_zone1_temp": "Machine2.Zone1Temperature.C.Actual",
            "machine2_zone2_temp": "Machine2.Zone2Temperature.C.Actual",
            "machine2_motor_amperage": "Machine2.MotorAmperage.U.Actual",
            "machine2_motor_rpm": "Machine2.MotorRPM.C.Actual",
            "machine2_material_pressure": "Machine2.MaterialPressure.U.Actual",
            "machine2_material_temp": "Machine2.MaterialTemperature.U.Actual",
            "machine2_exit_temp": "Machine2.ExitZoneTemperature.C.Actual",

            # Machine3 공정 변수
            "machine3_raw_property1": "Machine3.RawMaterial.Property1",
            "machine3_raw_property2": "Machine3.RawMaterial.Property2",
            "machine3_raw_property3": "Machine3.RawMaterial.Property3",
            "machine3_raw_property4": "Machine3.RawMaterial.Property4",
            "machine3_feeder_param": "Machine3.RawMaterialFeederParameter.U.Actual",
            "machine3_zone1_temp": "Machine3.Zone1Temperature.C.Actual",
            "machine3_zone2_temp": "Machine3.Zone2Temperature.C.Actual",
            "machine3_motor_amperage": "Machine3.MotorAmperage.U.Actual",
            "machine3_motor_rpm": "Machine3.MotorRPM.C.Actual",
            "machine3_material_pressure": "Machine3.MaterialPressure.U.Actual",
            "machine3_material_temp": "Machine3.MaterialTemperature.U.Actual",
            "machine3_exit_temp": "Machine3.ExitZoneTemperature.C.Actual",
        }

        self.variable_mapping.update(additional_mapping)

        # Stage1 측정값 3~14 추가 매핑 (세트포인트 포함)
        for idx in range(3, 15):
            actual_col = f"Stage1.Output.Measurement{idx}.U.Actual"
            setpoint_col = f"Stage1.Output.Measurement{idx}.U.Setpoint"
            self.variable_mapping[f"stage1_measurement{idx}"] = actual_col
            self.variable_mapping[f"stage1_measurement{idx}_setpoint"] = setpoint_col

        logger.info(f"DataProcessor 초기화 - CSV: {csv_path}")

    def load_data(self) -> pd.DataFrame:
        """데이터 로드"""
        logger.info("데이터 로딩 시작...")
        self.data = pd.read_csv(self.csv_path)
        logger.info(f"데이터 로드 완료 - Shape: {self.data.shape}")
        logger.info(f"결측치 확인: {self.data.isnull().sum().sum()}개")
        return self.data

    def create_mapped_dataset(self) -> pd.DataFrame:
        """매핑된 데이터셋 생성"""
        if self.data is None:
            self.load_data()

        # 매핑된 컬럼만 추출
        mapped_data = {}
        for friendly_name, original_col in self.variable_mapping.items():
            if original_col in self.data.columns:
                mapped_data[friendly_name] = self.data[original_col]
            else:
                logger.warning(f"컬럼 '{original_col}' 찾을 수 없음")

        df_mapped = pd.DataFrame(mapped_data)

        # 결측치 처리 (앞뒤 값으로 보간)
        df_mapped = df_mapped.ffill().bfill()

        logger.info(f"매핑된 데이터셋 생성 완료 - Shape: {df_mapped.shape}")
        logger.info(f"변수 목록: {list(df_mapped.columns)}")

        return df_mapped

    def prepare_ml_dataset(
        self,
        target_col: str = "welding_strength",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
        """
        ML 학습을 위한 데이터셋 준비

        Args:
            target_col: 예측 타겟 컬럼
            test_size: 테스트 세트 비율
            random_state: 랜덤 시드

        Returns:
            X_train, X_test, y_train, y_test, scaler
        """
        df_mapped = self.create_mapped_dataset()

        # Feature와 Target 분리
        feature_cols = [col for col in df_mapped.columns if col != target_col]
        X = df_mapped[feature_cols].values
        y = df_mapped[target_col].values

        logger.info(f"Feature 개수: {len(feature_cols)}")
        logger.info(f"Feature 목록: {feature_cols}")
        logger.info(f"Target: {target_col}")

        # Train/Test 분할 (스케일링 전) - 데이터 누수 방지
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        # 정규화 (0~1) - 학습 데이터 기준으로 Fit
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        logger.info(f"Train set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Target 범위: [{y.min():.2f}, {y.max():.2f}]")

        return X_train, X_test, y_train, y_test, self.scaler

    def save_scaler(self, filepath: str = "models/scaler.pkl"):
        """Scaler 저장"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler 저장 완료: {filepath}")

    def load_scaler(self, filepath: str = "models/scaler.pkl"):
        """Scaler 로드"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Scaler 로드 완료: {filepath}")
        return self.scaler

    def get_sample_data(self, idx: int = 0) -> Dict:
        """샘플 데이터 추출"""
        df_mapped = self.create_mapped_dataset()
        sample = df_mapped.iloc[idx].to_dict()
        return sample

    def create_anomaly_sample(self, base_idx: int = 0, anomaly_magnitude: float = 2.0) -> Dict:
        """
        이상 데이터 생성 (데모용)

        Args:
            base_idx: 베이스 데이터 인덱스
            anomaly_magnitude: 이상 크기 (표준편차 배수)

        Returns:
            이상 데이터 딕셔너리
        """
        sample = self.get_sample_data(base_idx)

        # press_thickness에 이상 주입
        df_mapped = self.create_mapped_dataset()
        thickness_std = df_mapped['press_thickness'].std()
        sample['press_thickness'] += anomaly_magnitude * thickness_std

        logger.info(f"이상 샘플 생성 - 두께 편차: +{anomaly_magnitude * thickness_std:.4f}")

        return sample


# 모듈 테스트
if __name__ == "__main__":
    logger.info("Data Preprocessing 테스트 시작")

    processor = ManufacturingDataProcessor()

    # 데이터 로드
    print("\n=== 데이터 로드 ===")
    df = processor.load_data()
    print(f"전체 컬럼 수: {len(df.columns)}")
    print(f"샘플 수: {len(df)}")

    # 매핑된 데이터셋 생성
    print("\n=== 변수 매핑 ===")
    df_mapped = processor.create_mapped_dataset()
    print(df_mapped.head())
    print(f"\n기술 통계:")
    print(df_mapped.describe())

    # ML 데이터셋 준비
    print("\n=== ML 데이터셋 준비 ===")
    X_train, X_test, y_train, y_test, scaler = processor.prepare_ml_dataset()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_train 범위: [{y_train.min():.2f}, {y_train.max():.2f}]")

    # Scaler 저장
    processor.save_scaler()

    # 샘플 데이터
    print("\n=== 샘플 데이터 ===")
    sample = processor.get_sample_data(0)
    for key, value in sample.items():
        print(f"{key}: {value:.2f}")

    # 이상 샘플
    print("\n=== 이상 샘플 생성 ===")
    anomaly_sample = processor.create_anomaly_sample(0, anomaly_magnitude=3.0)
    print(f"정상 두께: {sample['press_thickness']:.2f}")
    print(f"이상 두께: {anomaly_sample['press_thickness']:.2f}")

    logger.info("테스트 완료")
