Write a complete, modular Python script for automated feature selection and performance evaluation using Scikit-Learn that can work on any tabular regression or classification dataset.

Requirements:

1. Setup & Modularity:
   - Make the script generic so the user can easily swap in their own dataset path, target column name, task type ('regression' or 'classification'), and base estimator (e.g., LinearRegression, Ridge, RandomForest, or LogisticRegression).
   - Implement standard preprocessing: handle missing values, encode categorical variables, scale numerical features where appropriate, and create a train/test split (or cross-validation).

2. Feature Selection Scheme:
   - Provide options for systematic subset selection (e.g., SequentialFeatureSelector / Forward Selection, RFE, or evaluating top-k ranked features from k=1 up to n_features).
   - For each feature subset size k (from 1 to N features):
     * Fit the model on the selected k features.
     * Evaluate on test or validation data.
     * Store the number of features, the exact list of selected feature names, and primary evaluation metrics:
       - For Regression: RMSE / MSE and R-squared.
       - For Classification: Accuracy, F1-Score, and ROC-AUC (or Log Loss).

3. Output Table:
   - Print a clean, formatted comparison table (e.g., using pandas or tabulate) detailing:
     * Number of Features (k)
     * Selected Feature Names
     * Primary Error Metric (e.g., RMSE)
     * Primary Goodness-of-Fit Metric (e.g., R-squared)
   - Highlight the optimal subset based on best validation performance.

4. Visualization:
   - Generate two side-by-side plots:
     * Left: Error Metric vs. Number of Features (showing the minimum point / elbow).
     * Right: Performance/Fit Metric vs. Number of Features (showing the peak score).
   - Style the plots with clear titles, markers on data points, gridlines, and labeled axes.