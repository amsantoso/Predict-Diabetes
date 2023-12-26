import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import compute_class_weight


df = pd.read_csv('diabetes_prediction_dataset.csv')

# some data cleaning, factorizing and ordering
df['gender'] = df['gender'].astype('category')
gender_order = ['Female', 'Male', 'Other']
df['hypertension'] = df['hypertension'].astype('category')
hypertension_order = [0, 1] # 0= no, 1= yes
df['heart_disease'] = df['heart_disease'].astype('category')
heart_disease_order = [0, 1]  # 0= no, 1= yes
df['smoking_history'] = df['smoking_history'].astype('category')
smoking_history_order = ['No Info', 'current', 'ever', 'former', 'never', 'not current']
df['diabetes'] = df['diabetes'].astype('category')
diabetes_order = [0, 1]
def cont_distribution(var):
    # histogram
    plt.hist(df[var])
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.show()

    # box plot for outliers
    sns.boxplot(x=df[var])
    plt.title(f'Box plot for {var}')
    plt.show()

# subplot
plt.figure(figsize=(16, 12))
cont_distribution('age')
plt.show()
cont_distribution('bmi')
plt.show()
cont_distribution('HbA1c_level')
plt.show()
# check Multicollinearity
corr_matrix = df[['age', 'blood_glucose_level', 'HbA1c_level', 'bmi']]
sns.heatmap(corr_matrix.corr(), annot=True)
plt.title('Correlation Matrix')
plt.show()

# for categorical variables
sns.countplot(df, x='gender', order=gender_order).set_title('Gender Distribution')
sns.countplot(df, x='hypertension', order=hypertension_order).set_title('Hypertension Distribution')
sns.countplot(df, x='heart_disease', order=heart_disease_order).set_title('Heart Disease Distribution')
sns.countplot(df, x='smoking_history', order=smoking_history_order).set_title('Smoking History Distribution')
sns.countplot(df, x='diabetes', order=diabetes_order).set_title('Diabetes Distribution')

# test and train the data set with algorithms: gradient boost, random forest and XGBoost
X = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
y = df['diabetes']
X = pd.get_dummies(X, columns=['gender', 'smoking_history'], drop_first=True) # one hot encode
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df['binary_smoking_history'] = df['smoking_history'].apply(lambda x: 0 if x in ['no info', 'not current', 'never'] else 1) # binary labels for smoking history to use in class weights

# class weights
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
sample_weights = class_weights[y_train.astype(int)]

# gradient boost
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train, sample_weight=sample_weights)
gb_pred = gb_model.predict(X_test) # predictions
gb_cf = confusion_matrix(y_test, gb_pred)
gb_accuracy = accuracy_score(y_test, gb_pred)

# random forest
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test) # predictions
rf_cf = confusion_matrix(y_test, rf_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

#XGBoost
xgb_model = XGBClassifier(random_state=42, scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), enable_categorical=True)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_cf = confusion_matrix(y_test,xgb_pred)
xgb_accuracy = accuracy_score(y_test,xgb_pred)


# print summary
print("Gradient Boosting Classifier:")
print(f"Confusion Matrix:\n{gb_cf}")
print(f"Accuracy: {gb_accuracy:.2f}")

print("\nRandom Forest Classifier:")
print(f"Confusion Matrix:\n{rf_cf}")
print(f"Accuracy: {rf_accuracy:.2f}")

print("\nXGBoost Classifier:")
print(f"Confusion Matrix:\n{xgb_cf}")
print(f"Accuracy: {xgb_accuracy:.2f}")