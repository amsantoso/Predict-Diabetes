This python code explores and uses the data set diabetes_prediction_dataset.csv from kaggle (https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) to find and build the best suited machine learning model for predicting the risk of diabetes in individuals. The dataset includes key health-related features such as blood glucose levels, A1C test results, gender, age, hypertension, heart disease, smoking history, BMI, and the diabetes label.

The project started with exploratory data analysis, visualizing the distribution of variables, checking for multicollinearity, and identifying outliers. Then it preprocesses the data, converting categorical variables into dummy/indicator variables, and addressing class imbalance in certain features.

For modeling, Gradient Boost, Random Forest, and XGBoost were algorithms chosen to see which algorithm is best to predict diabetes risk based on the selected features. To see which model performed best, they were evaluated using accuracy and a confusion matrix. Additionally, class weights are incorporated to address imbalances in the data set.

