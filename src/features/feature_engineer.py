"""
Feature Engineering 모듈

데이터셋에 독립적인 피처 생성 로직
"""
import pandas as pd
import numpy as np
from typing import Callable, List, Dict, Optional
from dataclasses import dataclass
from loguru import logger

from config.data_schema import DataSchema


@dataclass
class FeatureRecipe:
    """피처 생성 레시피"""
    name: str
    func: Callable[[pd.DataFrame], pd.Series]
    description: str = ""
    dependencies: List[str] = None  # 필요한 컬럼들

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class FeatureEngineer:
    """
    Feature Engineering 엔진

    데이터셋 스키마를 기반으로 피처를 동적으로 생성
    """

    def __init__(self, schema: DataSchema):
        """
        Args:
            schema: 데이터셋 스키마
        """
        self.schema = schema
        self.recipes: List[FeatureRecipe] = []

        # 기본 레시피 등록
        self._register_default_recipes()

    def add_recipe(self, recipe: FeatureRecipe):
        """피처 레시피 추가"""
        self.recipes.append(recipe)
        logger.debug(f"레시피 추가: {recipe.name}")

    def apply(self, df: pd.DataFrame, skip_missing: bool = True) -> pd.DataFrame:
        """
        모든 피처 생성

        Args:
            df: 원본 데이터프레임
            skip_missing: 필요한 컬럼이 없으면 스킵 (True) or 에러 (False)

        Returns:
            피처가 추가된 데이터프레임
        """
        df_fe = df.copy()

        for recipe in self.recipes:
            # 의존성 체크
            missing_deps = [d for d in recipe.dependencies if d not in df_fe.columns]

            if missing_deps:
                if skip_missing:
                    logger.warning(
                        f"피처 '{recipe.name}' 스킵: 필요한 컬럼 없음 {missing_deps}"
                    )
                    continue
                else:
                    raise ValueError(
                        f"피처 '{recipe.name}' 생성 실패: 필요한 컬럼 없음 {missing_deps}"
                    )

            try:
                df_fe[recipe.name] = recipe.func(df_fe)
                logger.debug(f"피처 생성: {recipe.name}")
            except Exception as e:
                logger.error(f"피처 '{recipe.name}' 생성 중 오류: {e}")
                if not skip_missing:
                    raise

        return df_fe

    def recalculate_features(
        self,
        row: Dict[str, float],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        특정 피처들만 재계산 (조정 후 사용)

        Args:
            row: 데이터 딕셔너리
            feature_names: 재계산할 피처 이름들 (None이면 스키마의 recalculable_features)

        Returns:
            업데이트된 데이터 딕셔너리
        """
        if feature_names is None:
            feature_names = self.schema.recalculable_features

        updated = row.copy()

        # DataFrame 형태로 변환 (레시피 함수가 DataFrame 기대)
        df_temp = pd.DataFrame([updated])

        for recipe in self.recipes:
            if recipe.name not in feature_names:
                continue

            # 의존성 체크
            missing_deps = [d for d in recipe.dependencies if d not in updated]
            if missing_deps:
                logger.warning(
                    f"피처 '{recipe.name}' 재계산 스킵: 필요한 컬럼 없음 {missing_deps}"
                )
                continue

            try:
                result = recipe.func(df_temp)
                updated[recipe.name] = float(result.iloc[0])
                logger.debug(f"피처 재계산: {recipe.name} = {updated[recipe.name]:.4f}")
            except Exception as e:
                logger.warning(f"피처 '{recipe.name}' 재계산 실패: {e}")

        return updated

    def _register_default_recipes(self):
        """
        기본 피처 레시피 등록

        continuous_factory 데이터셋 기반 레시피
        """
        # 1. heat_input_proxy
        self.add_recipe(FeatureRecipe(
            name="heat_input_proxy",
            func=lambda df: df['welding_temp1'] / (df['welding_temp3'] + 1e-5),
            description="Heat input proxy (temp1 / temp3)",
            dependencies=["welding_temp1", "welding_temp3"]
        ))

        # 2. pressure_x_temp2
        self.add_recipe(FeatureRecipe(
            name="pressure_x_temp2",
            func=lambda df: df['welding_pressure'] * df['welding_temp2'],
            description="Pressure × Temperature interaction",
            dependencies=["welding_pressure", "welding_temp2"]
        ))

        # 3. total_control
        self.add_recipe(FeatureRecipe(
            name="total_control",
            func=lambda df: df['welding_control1'] + df['welding_control2'],
            description="Sum of control variables",
            dependencies=["welding_control1", "welding_control2"]
        ))

        # 4. press_volume_proxy
        self.add_recipe(FeatureRecipe(
            name="press_volume_proxy",
            func=lambda df: df['press_thickness'] * df['press_measurement1'],
            description="Press volume proxy",
            dependencies=["press_thickness", "press_measurement1"]
        ))

        # 5. welding_temp_mean
        self.add_recipe(FeatureRecipe(
            name="welding_temp_mean",
            func=lambda df: df[['welding_temp1', 'welding_temp2', 'welding_temp3',
                                'welding_temp4', 'welding_temp5']].mean(axis=1),
            description="Mean of welding temperatures",
            dependencies=["welding_temp1", "welding_temp2", "welding_temp3",
                         "welding_temp4", "welding_temp5"]
        ))

        # 6. welding_temp_span
        self.add_recipe(FeatureRecipe(
            name="welding_temp_span",
            func=lambda df: (
                df[['welding_temp1', 'welding_temp2', 'welding_temp3',
                    'welding_temp4', 'welding_temp5']].max(axis=1) -
                df[['welding_temp1', 'welding_temp2', 'welding_temp3',
                    'welding_temp4', 'welding_temp5']].min(axis=1)
            ),
            description="Range of welding temperatures",
            dependencies=["welding_temp1", "welding_temp2", "welding_temp3",
                         "welding_temp4", "welding_temp5"]
        ))

    def add_setpoint_error_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Setpoint Error 및 Ratio 피처 추가

        Args:
            df: 데이터프레임
            target_col: 타겟 컬럼 (제외할 컬럼)

        Returns:
            피처가 추가된 데이터프레임
        """
        df_fe = df.copy()

        if target_col is None:
            target_col = self.schema.target_variable

        setpoint_cols = [col for col in df_fe.columns if col.endswith('_setpoint')]

        for setpoint_col in setpoint_cols:
            actual_col = setpoint_col.replace('_setpoint', '')

            # 타겟 유출 방지
            if actual_col == target_col:
                continue

            if actual_col not in df_fe.columns:
                continue

            # Error 피처
            error_col = f"{actual_col}_error"
            df_fe[error_col] = df_fe[actual_col] - df_fe[setpoint_col]

            # Ratio 피처
            ratio_col = f"{actual_col}_ratio"
            df_fe[ratio_col] = np.where(
                df_fe[setpoint_col] != 0,
                df_fe[actual_col] / df_fe[setpoint_col],
                1.0
            )

        return df_fe

    def add_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        집계 통계 피처 추가 (Stage1 측정치 등)

        Args:
            df: 데이터프레임

        Returns:
            피처가 추가된 데이터프레임
        """
        df_fe = df.copy()

        # Stage1 error 집계
        stage1_error_cols = [
            col for col in df_fe.columns
            if col.startswith('stage1_measurement') and col.endswith('_error')
        ]

        if stage1_error_cols:
            stage_error_df = df_fe[stage1_error_cols]
            df_fe['stage1_error_mean'] = stage_error_df.mean(axis=1)
            df_fe['stage1_error_std'] = stage_error_df.std(axis=1).fillna(0)
            df_fe['stage1_error_abs_max'] = stage_error_df.abs().max(axis=1)

        # Stage1 ratio 집계
        stage1_ratio_cols = [
            col for col in df_fe.columns
            if col.startswith('stage1_measurement') and col.endswith('_ratio')
        ]

        if stage1_ratio_cols:
            stage_ratio_df = df_fe[stage1_ratio_cols]
            df_fe['stage1_ratio_mean'] = stage_ratio_df.mean(axis=1)
            df_fe['stage1_ratio_std'] = stage_ratio_df.std(axis=1).fillna(0)

        return df_fe
