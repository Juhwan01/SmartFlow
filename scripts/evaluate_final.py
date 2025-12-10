"""
ML 모델 최종 평가 스크립트 (Test Set)

WARNING: 이 스크립트는 단 1회만 실행하세요!
Test 데이터로 모델의 최종 성능을 검증합니다.
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
from loguru import logger
import json
from datetime import datetime


def calculate_mape(y_true, y_pred):
    """MAPE 계산"""
    threshold = 0.1
    mask = np.abs(y_true) > threshold
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("TEST SET 최종 평가 (ML 모델)")
    print("=" * 70)
    print("WARNING: 이 평가는 단 1회만 실행되어야 합니다!")
    print("=" * 70)

    try:
        # 1. Test 데이터 로드
        test_path = Path("data/test_set.csv")
        if not test_path.exists():
            logger.error(f"Test 데이터가 없습니다: {test_path}")
            logger.error("먼저 'python scripts/train_model.py'를 실행하세요.")
            sys.exit(1)

        test_df = pd.read_csv(test_path)
        logger.info(f"Test 데이터 로드: {test_path} ({len(test_df)} samples)")

        # 2. 피처와 타겟 분리
        target_col = "welding_strength"
        feature_cols = [col for col in test_df.columns if col != target_col]

        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values

        logger.info(f"Feature 수: {len(feature_cols)}")
        logger.info(f"Target: {target_col}")

        # 3. 모델 로드
        model_path = Path("models/quality_predictor.pkl")
        if not model_path.exists():
            logger.error(f"모델 파일이 없습니다: {model_path}")
            sys.exit(1)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"모델 로드: {model_path}")

        # 4. 예측
        logger.info("Test 데이터 예측 중...")
        y_pred = model.predict(X_test)

        # 5. 성능 평가
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = calculate_mape(y_test, y_pred)

        # 6. 결과 출력
        print("\n" + "=" * 70)
        print("TEST SET 최종 성능")
        print("=" * 70)
        print(f"MAE (평균 절대 오차)     : {mae:.4f} (목표: < 0.2)")
        print(f"RMSE (제곱근 평균 오차)  : {rmse:.4f}")
        print(f"MAPE (평균 오차율)       : {mape:.2f}% (목표: < 2.0%)")
        print("=" * 70)

        # 7. 목표 달성 여부 확인
        mae_pass = bool(mae < 0.2)
        mape_pass = bool(mape < 2.0)

        print("\n목표 달성 여부:")
        print(f"  MAE  < 0.2   : {'PASS' if mae_pass else 'FAIL'}")
        print(f"  MAPE < 2.0%  : {'PASS' if mape_pass else 'FAIL'}")
        print("=" * 70)

        # 8. 결과 저장
        results = {
            "evaluation_date": datetime.now().isoformat(),
            "test_samples": len(y_test),
            "metrics": {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "MAPE": float(mape)
            },
            "goals": {
                "MAE_target": 0.2,
                "MAE_achieved": mae_pass,
                "MAPE_target": 2.0,
                "MAPE_achieved": mape_pass
            },
            "overall_pass": mae_pass and mape_pass
        }

        output_dir = Path("models")
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "final_test_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nTest 결과 저장: {results_path}")

        # 9. 텍스트 리포트 저장
        report_path = output_dir / "test_evaluation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("SmartFlow ML 모델 최종 평가 리포트 (Test Set)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"평가 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test 샘플 수: {len(y_test)}\n\n")

            f.write("=" * 70 + "\n")
            f.write("성능 지표\n")
            f.write("=" * 70 + "\n")
            f.write(f"MAE (평균 절대 오차)     : {mae:.4f} (목표: < 0.2)\n")
            f.write(f"RMSE (제곱근 평균 오차)  : {rmse:.4f}\n")
            f.write(f"MAPE (평균 오차율)       : {mape:.2f}% (목표: < 2.0%)\n\n")

            f.write("=" * 70 + "\n")
            f.write("목표 달성 여부\n")
            f.write("=" * 70 + "\n")
            f.write(f"MAE  < 0.2   : {'PASS' if mae_pass else 'FAIL'}\n")
            f.write(f"MAPE < 2.0%  : {'PASS' if mape_pass else 'FAIL'}\n\n")

            if results["overall_pass"]:
                f.write("=" * 70 + "\n")
                f.write("종합 평가: PASS - 현장 투입 가능!\n")
                f.write("=" * 70 + "\n")
            else:
                f.write("=" * 70 + "\n")
                f.write("종합 평가: FAIL - 모델 개선 필요\n")
                f.write("=" * 70 + "\n")

        logger.info(f"평가 리포트 저장: {report_path}")

        # 10. 최종 메시지
        print("\n" + "=" * 70)
        if results["overall_pass"]:
            print("최종 평가: PASS")
            print("모든 목표를 달성했습니다! 현장 투입 가능한 모델입니다.")
        else:
            print("최종 평가: FAIL")
            print("일부 목표를 달성하지 못했습니다. 모델 개선이 필요합니다.")
        print("=" * 70)
        print("\n결과 파일:")
        print(f"  - {results_path}")
        print(f"  - {report_path}")
        print("=" * 70)

    except Exception as e:
        logger.exception(f"평가 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
