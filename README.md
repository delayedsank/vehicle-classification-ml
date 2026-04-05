

# Vehicle Classification with Machine Learning

Multi-class vehicle classification on the Statlog Vehicle Silhouette dataset 
(4 classes: Bus, Opel, Saab, Van) using shape features from low-resolution imagery.

## Models Evaluated
| Model               | Accuracy |
|---------------------|----------|
| Decision Tree       | 74.2%    |
| Random Forest       | ~79%     |
| Gradient Boosting   | ~79%     |
| Logistic Regression | 80.66%   |
| Linear SVM          | 80.0%    |

## Key Findings
- Logistic Regression outperformed Decision Tree by ~6.5%
- Main confusion: Opel vs Saab (visually similar silhouettes)
- Bus class achieved near-perfect classification (F1: 0.97)

## Stack
Python, scikit-learn, pandas, NumPy, Matplotlib, Seaborn

## Report
See `vehicle_classfication_report.pdf` for full analysis including 
confusion matrices, per-class precision/recall/F1, and methodology.
