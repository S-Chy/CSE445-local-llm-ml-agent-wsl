# Model Comparison Benchmark

Generated: 2026-09-06 21:05

3 algorithms (decision_tree, logistic_regression, random_forest) evaluated across 2 datasets (wine, breast_cancer) using 5-fold cross-validation.

## Results

| Dataset | Model | Test Accuracy | CV Mean Accuracy | CV Std |
|---|---|---|---|---|
| wine | decision_tree | 0.9444 | 0.8937 | 0.0472 |
| wine | logistic_regression | 0.9722 | 0.9611 | 0.0333 |
| wine | random_forest | 1.0 | 0.961 | 0.0221 |
| breast_cancer | decision_tree | 0.9386 | 0.9209 | 0.0202 |
| breast_cancer | logistic_regression | 0.9561 | 0.9543 | 0.0102 |
| breast_cancer | random_forest | 0.9561 | 0.9543 | 0.0244 |

## Best model per dataset

- **wine**: best model is `logistic_regression` (CV mean accuracy 0.9611, std 0.0333)
- **breast_cancer**: best model is `logistic_regression` (CV mean accuracy 0.9543, std 0.0102)
