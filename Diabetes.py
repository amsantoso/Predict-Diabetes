import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


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

# explanatory data analysis
# show distribution of continuous variables: age, bmi, HbA1c_level, blood_glucose_level
# using histogram
def cont_distribution(var, pos):
    # histogram
    plt.subplot(pos)
    plt.hist(df[var])
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.show()

    # box plot for outliers
    plt.subplot(pos)
    sns.boxplot(x=df[var])
    plt.title(f'Box plot for {var}')
    plt.show()

# subplot
plt.figure(figsize=(16, 12))

continuous_variables = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

for i, variable in enumerate(continuous_variables, start=1):
    cont_distribution(variable, 220 + i)

plt.tight_layout()
plt.show() # show the distribution of the variables

# check Multicollinearity
corr_matrix = df[['age', 'blood_glucose_level', 'HbA1c_level', 'bmi']]
sns.heatmap(corr_matrix.corr(), annot=True)
plt.title('Correlation Matrix')
plt.show()
# comment: No Multicollinearity, suggests a relatively low to moderate correlation between these variables.
# comment: Not much outliers

# Show distribution of categorical variables: gender, HTN, heart_disease, smoking_history, diabetes
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

sns.countplot(df, x='gender', ax=axes[0, 0], order=gender_order)
axes[0, 0].set_title('Gender Distribution')
sns.countplot(df, x='hypertension', ax=axes[0, 1], order=hypertension_order)
axes[0, 1].set_title('hypertension Distribution')
sns.countplot(df, x='heart_disease', ax=axes[0, 2], order=heart_disease_order)
axes[0, 2].set_title('heart_disease Distribution')
sns.countplot(df, x='smoking_history', ax=axes[1, 0], order=smoking_history_order)
axes[1, 0].set_title('smoking_history Distribution')
sns.countplot(df, x='diabetes', ax=axes[1, 1], order=diabetes_order)
axes[1, 1].set_title('diabetes Distribution')

plt.tight_layout()
plt.show()

# I chose random forest to test and train my dataset with
# Split the data into training and testing; 20% test, 80% train
X = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
y = df['diabetes']
X = pd.get_dummies(X, columns=['gender', 'smoking_history'], drop_first=True) # one hot encode

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
# predictions
y_pred = rf_model.predict(X_test)
# check goodness of fit
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
# confusion matrix
rf_cf = confusion_matrix(y_test, y_pred)
print(rf_cf)