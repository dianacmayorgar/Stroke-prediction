import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Choose a dataset with at least 20,000 observations
df = pd.read_csv('train_2v.csv')
df

#Perform initial exploration to understand the data structure, types of variables, and identify errors or missing values.
df.info()

df.describe()

# Identifying and handling missing values.
df.isnull().sum()

# Duplication Removal
df.drop_duplicates()

"""df['bmi'].fillna(df['bmi'].median(), inplace= True)"""

# Checking the 'bmi' distributions
"""sns.histplot(df['bmi'])"""

df['bmi'] = df.groupby('age')['bmi'].transform(lambda x: x.fillna(x.mean()))

# Checking the 'bmi' distributions
sns.histplot(df['bmi'])

# Removing records where smoking status is null value
df.dropna(subset=['smoking_status'], inplace=True)

df[(df['work_type']== 'children') & (df['stroke']== 1)]

df.reset_index(drop=True, inplace = True)


df

#0.5 marks: Identifying errors or inconsistencies in the data (Typo errors)
df['gender'].unique()

df['gender'].value_counts()

# Only a few observation were labelled as others and due to there was no information on these patients,
# they were excluded for the reamining of the analysis.
df = df[df['gender'].isin(['Female', 'Male'])]

df['work_type'].unique()

df['Residence_type'].unique()

df['smoking_status'].unique()

df['ever_married'].unique()

# Update the 'hypertension' column
df['hypertension'] = df['hypertension'].apply(lambda x: 'no hypertension' if x == 0 else 'hypertension')

# Update the 'heart_disease' column
df['heart_disease'] = df['heart_disease'].apply(lambda x: 'no heart disease' if x == 0 else 'heart disease')

# Update the 'stroke' column
df['stroke'] = df['stroke'].apply(lambda x: 'no stroke' if x == 0 else 'stroke')

q10, q25, q50, q75= df['age'].quantile([0.10, 0.25, 0.50, 0.75]).values
Iqr=q75-q25
lf=int(q25-1.5*Iqr)
uf=int(q75+1.5*Iqr)
type(uf)

range_outlier_age = tuple([lf, uf])
range_outlier_age

df['age'].describe()

out_age = df['age'][uf<df['age']]
out_age

q10, q25, q50, q75= df['avg_glucose_level'].quantile([0.10, 0.25, 0.50, 0.75]).values
Iqr=q75-q25
lf=int(q25-2.0*Iqr)
uf=int(q75+2.0*Iqr)

range_outlier_glucosa = tuple([lf, uf])
range_outlier_glucosa

out_glu = df['avg_glucose_level'][uf<df['avg_glucose_level']]
out_glu

q10, q25, q50, q75= df['bmi'].quantile([0.10, 0.25, 0.50, 0.75]).values
Iqr=q75-q25
lf=int(q25-2.0*Iqr)
uf=int(q75+2.0*Iqr)


range_outlier_bmi = tuple([lf, uf])
range_outlier_bmi

out_bmi = df['bmi'][uf<df['bmi']].sort_values(ascending=False)
out_bmi

# Remove or impute outliers if necessary categorical.
numerical= ['avg_glucose_level', 'bmi', 'age']

plt.figure(figsize=(15,8))
for i, j in enumerate(numerical):
    plt.subplot(1,3, i+1)
    sns.boxplot(x =df['stroke'],y = df[j] )
    plt.title(f'stroke vs {j}')

plt.xticks(rotation=90)

plt.show()

df.info()

string_columns =['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke']
for i in string_columns:
  df[i] = df[i].astype('string')

df.info()

string_columns =['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke']
for i in string_columns:
  df[i] = df[i].str.strip().str.lower()

df

#Removing useless column
df.drop(['id'],inplace=True, axis=1)

df

counts_df = len(df)

# Groupy by 'gender' and 'stroke'
result_2 = df.groupby(['gender', 'stroke']).agg(
    n=('gender', 'size'),
    percentage=('gender', lambda x: (x.size / counts_df) * 100),
    age_mean=('age', 'mean'),
    age_sd=('age', 'std')
).reset_index()

result_2



#Checking the distribution of strokes and smoking status
counts_df = len(df)

# Group by 'gender' and 'stroke'
result_2 = df.groupby(['smoking_status', 'stroke']).agg(
    n=('smoking_status', 'size'),
    percentage=('smoking_status', lambda x: (x.size / counts_df) * 100),

).reset_index()

result_2


#Creating New Feature EverSmoked
df['ever_smoked'] = df['smoking_status'].apply(lambda x: 'Smoked' if x in ['smokes', 'formerly smoked'] else 'no smoked')

counts_df = len(df)

# Grouping by 'ever_smoked' and  'stroke'
result_2 = df.groupby(['ever_smoked', 'stroke']).agg(
    n=('ever_smoked', 'size'),
    percentage=('ever_smoked', lambda x: (x.size / counts_df) * 100),

).reset_index()

result_2





df2 = df.copy()

df

df2

"""df2.to_csv('train_2v-clean.csv', index = False)"""

df.info()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Fit and transform the training data
columns = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'ever_smoked', 'stroke']
for i in columns:
  df[i + '_encoded'] = le.fit_transform(df[i])




#Dropping this columns 'gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'ever_smoked', 'stroke'
for i in columns:
  df.drop([i], axis = 1, inplace =True)

df



df3 = df2.copy()
for column in columns:
    frequency_encoding = df3[column].value_counts(normalize=True)
    df3[column + '_encoded'] = df3[column].map(frequency_encoding)

for i in columns:
  df3.drop([i], axis = 1, inplace =True)

df3.reset_index(drop = True)
df3.head()

#due that the dataset is unbalanced the correlaion is not visible
plt.figure(figsize= (15,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_cols =['age','avg_glucose_level','bmi']

# Fit and transform the training data
df_scaled = scaler.fit_transform(df[numeric_cols])
df_scaled_2 = pd.DataFrame(df_scaled, columns=numeric_cols)


#Consistency in scaling across the dataset we reset index to assure that the number of rows is equal in both dataset
df_scaled_2.reset_index(drop =True, inplace = True)

df_scaled_final = pd.concat([df.drop(['age','avg_glucose_level','bmi'], axis=1), df_scaled_2], axis=1)
df_scaled_final

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_cols =['age','avg_glucose_level','bmi']

# Fit and transform the training data
df_scaled = scaler.fit_transform(df[numeric_cols])
df_scaled_2 = pd.DataFrame(df_scaled, columns=numeric_cols)


#Consistency in scaling across the dataset we reset index to assure that the number of rows is equal in both dataframe
df_scaled_2.reset_index(drop =True, inplace = True)

df_scaled_final = pd.concat([df.drop(['age','avg_glucose_level','bmi'], axis=1).reset_index(drop =True), df_scaled_2], axis=1)
df_scaled_final

df.head()

X = df_scaled_final.drop('stroke_encoded', axis=1)
y = df_scaled_final['stroke_encoded']

#train_test_split is used with the stratify=y parameter, which ensures that the class proportions in y are preserved in the train and test sets.
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30, stratify=y,random_state=42)

X_train.shape ,X_test.shape,y_train.shape,y_test.shape


# Set the style of the plot to whitegrid
sns.set_style("whitegrid")

# Set the color palette to "husl"
sns.set_palette("husl")

# Create a new figure with a specified figure size
plt.figure(figsize=(15, 10))

# Define the list of continuous columns
continous_data = ["age", "avg_glucose_level", "bmi"]

# Iterate through each continuous column
for i, column in enumerate(continous_data, 1):
    # Create a subplot for the current continuous column
    plt.subplot(len(continous_data), 1, i)

    # Create a histogram plot for the current continuous column with kernel density estimation (kde) and density statistics (stat="density")
    sns.histplot(df2[column], kde=True, stat="density")

    # Add vertical lines for the mean and median of the data
    plt.axvline(df2[column].mean(), color='r', linestyle='--', label='Mean')
    plt.axvline(df2[column].median(), color='g', linestyle='-', label='Median')

    # Add a legend to the plot
    plt.legend()

    # Set the title, x-axis label, and y-axis label for the plot
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Density")

    # Adjust the layout of subplots
    plt.tight_layout()

# Define the path to save the PDF file
#pdf_path_continuous = "continuous_variables2.pdf"

# Save the plot as a PDF file
#plt.savefig(pdf_path_continuous)

# Display the plot
plt.show()


df2



sns.set_style("whitegrid")
colors = sns.color_palette("pastel")
plt.figure(figsize=(12, 8))
categorical_data = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "ever_smoked", "stroke"]

for i, column in enumerate(categorical_data):
    plt.subplot(2, 4, i+1)
    ax = sns.countplot(x=column, data=df2, palette=colors)

    plt.title(f"Distribution of {column}")
    plt.xlabel("")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    # Add count values on top of each bar
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2., p.get_height(), f'{int(p.get_height())}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()


# Display the plot
plt.show()



sns.pairplot(data=df2)

df2

df_age_stroke = df2[['stroke', 'age']]

# Create a density plot using Seaborn
plt.figure(figsize=(10, 6))  # Set the size of the figure
sns.kdeplot(data=df_age_stroke, x='age', hue='stroke', fill=True, palette={'no stroke': 'skyblue', 'stroke': 'orange'}, common_norm=False)
# Plot the density of 'age' with different colors for 'stroke' categories, fill the area under the curves

# Use skyblue color for 'no stroke' and orange color for 'stroke'
plt.title('Age Distribution by Stroke and Age.')  # Set the title of the plot
plt.xlabel('Age')  # Set the label for the x-axis
plt.ylabel('Density')  # Set the label for the y-axis

plt.legend(title='Stroke')  # Add a legend with the title 'Stroke'
 # Display the plot

plt.show()

# Get the counts of strokes and smoking status from the oversampled DataFrame

# Sort the data by smoking status and the presence of strokes
#stroke_smoking_counts = stroke_smoking_counts.sort_values(by=['ever_smoked', 'stroke'])

stroke_smoking_counts = df2.groupby(['ever_smoked'])['stroke'].value_counts().reset_index()

# Create the stacked bar plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='ever_smoked', y='count', hue='stroke', data=stroke_smoking_counts, palette={'stroke': 'skyblue', 'no stroke': 'orange'})
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2., p.get_height(), f'{int(p.get_height())}',
            ha='center', va='bottom', fontsize=10)

plt.title('Relationship between Strokes and Smoking Status')
plt.xlabel('Smoking Status')
plt.ylabel('Number of Cases')
plt.xticks(rotation=45)
plt.legend(title='Stroke')
plt.show()



df_avg_glucose_stroke = df2[['stroke', 'avg_glucose_level']].reset_index()

# Create the violin plot using Seaborn
plt.figure(figsize=(10, 6))
sns.violinplot(x='stroke', y='avg_glucose_level', data=df_avg_glucose_stroke, palette={'no stroke': 'skyblue', 'stroke': 'orange'})
plt.title('Relationship between Strokes and Average Blood Glucose Level')
plt.xlabel('Stroke')
plt.ylabel('Average Blood Glucose Level')
plt.savefig("Relationship between Strokes and Average Blood Glucose Level", dpi=300)
plt.show()


df_scaled_final.groupby('stroke_encoded').count().reset_index()

df_scaled_final

import pandas as pd
from sklearn.utils import resample

# Separate positive and negative examples
df_positive = df_scaled_final[df_scaled_final['stroke_encoded'] == 1]
df_negative = df_scaled_final[df_scaled_final['stroke_encoded'] == 0]

# Calculating the minimum number of examples (under-sampling to the majority group)
n_samples = min(len(df_positive), len(df_negative))

# Subsampling negative examples to equal the number of positive examples
df_negative_resampled = resample(df_negative, replace=False, n_samples=n_samples, random_state=42)

# Merging subsampled DataFrames
df_balanced = pd.concat([df_positive, df_negative_resampled])

# Optional: Shuffle the resulting DataFrame
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
df_balanced


# df_balanced now contains a subsampled DataFrame with balance between stroke classes


df_balanced.groupby('stroke_encoded').count().reset_index()


plt.figure(figsize = (15,8))
sns.heatmap(df_balanced.corr(), annot=True, cmap ='coolwarm')

from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

# Initializing the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# Initializing RandomUnderSampler
under_sampler = RandomUnderSampler(random_state=42)

# Performing under-sampling on the training data
X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train, y_train)

# Training the model with the training data
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy del modelo: {accuracy:.2f}')
confusion_matrix(y_test, y_pred)

# Show additional metrics (optional)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(precision_score(y_test, y_pred))




sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='coolwarm', annot_kws={"size": 10} )
plt.xlabel('Predicted')
plt.ylabel('Actual')

from imblearn.over_sampling import RandomOverSampler

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# Initialize RandomOverSampler
over_sampler = RandomOverSampler(random_state=42)

# Perform oversampling on the training data
X_train_resampled, y_train_resampled = over_sampler.fit_resample(X_train, y_train)

# Train the model with the oversampled data
model.fit(X_train_resampled, y_train_resampled)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy del modelo: {accuracy:.2f}')

# Print confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Print classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Create a DataFrame to compare actual vs predicted
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results_df.head())


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='coolwarm', annot_kws={"size": 10}, fmt='d' )
plt.xlabel('Predicted')
plt.ylabel('Actual')

from sklearn.naive_bayes import GaussianNB

# Initializing the Naive Bayes model
model = GaussianNB()

# Initializing RandomUnderSampler
under_sampler = RandomUnderSampler(random_state=42)

# Performing under-sampling on the training data
X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train, y_train)

# Training the model with the training data
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model: {accuracy:.2f}')

# Displaying the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Showing additional metrics
print(classification_report(y_test, y_pred))


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='coolwarm', annot_kws={"size": 10}, fmt='d' )
plt.xlabel('Predicted')
plt.ylabel('Actual')

from imblearn.over_sampling import RandomOverSampler

# Initializing the Naive Bayes model
model = GaussianNB()

# Initialize RandomOverSampler
over_sampler = RandomOverSampler(random_state=42)

# Perform oversampling on the training data
X_train_resampled, y_train_resampled = over_sampler.fit_resample(X_train, y_train)


# Train the model with the oversampled data
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model: {accuracy:.2f}')

# Displaying the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Showing additional metrics
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='coolwarm', annot_kws={"size": 10}, fmt='d' )
plt.xlabel('Predicted')
plt.ylabel('Actual')

from sklearn.svm import SVC

# Initialize RandomOverSampler
over_sampler = RandomOverSampler(random_state=42)

# Perform oversampling on the training data
X_train_resampled, y_train_resampled = over_sampler.fit_resample(X_train, y_train)

# Training de model SVC
model = SVC(kernel='linear', random_state=42)

# Train the model with the oversampled data
model.fit(X_train_resampled, y_train_resampled)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='coolwarm', annot_kws={"size": 10}, fmt='d' )
plt.xlabel('Predicted')
plt.ylabel('Actual')

from sklearn.svm import SVC

# Initializing RandomUnderSampler
under_sampler = RandomUnderSampler(random_state=42)

# Performing under-sampling on the training data
X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train, y_train)


# Training the model SVC
model = SVC(kernel='linear', random_state=42)


# Training the model with the training data
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='coolwarm', annot_kws={"size": 10}, fmt='d' )
plt.xlabel('Predicted')
plt.ylabel('Actual')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

under_sampler = RandomUnderSampler(random_state=42)

# Performing under-sampling on the training data
X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train, y_train)

# Define the parameter grid to tune the model
param_grid = {
    'max_depth': [None, 10, 13, 17, 20, 23, 27, 30],
    'min_samples_split': [2, 5, 10, 13, 17, 20],
    'min_samples_leaf': [1, 3, 5, 7, 10, 15]
}

# Setting up the GridSearchCV objects, one for each criterion
grid_search_gini = GridSearchCV(
    DecisionTreeClassifier(criterion='gini', random_state=42),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy'
)

# Fitting the GridSearchCV objects
grid_search_gini.fit(X_train_resampled, y_train_resampled)
y_pred = grid_search_gini.predict(X_test)

# Getting the best parameters and best scores
best_params_gini = grid_search_gini.best_params_
best_score_gini = grid_search_gini.best_score_

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model: {accuracy:.2f}')

# Displaying the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Showing additional metrics
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='coolwarm', annot_kws={"size": 10}, fmt='d' )
plt.xlabel('Predicted')
plt.ylabel('Actual')



grid_search_entropy = GridSearchCV(
    DecisionTreeClassifier(criterion='entropy', random_state=42),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='recall'
)

# Fitting the GridSearchCV objects
grid_search_entropy.fit(X_train_resampled, y_train_resampled)
y_pred = grid_search_entropy.predict(X_test)

# Getting the best parameters and best scores
best_params_entropy = grid_search_entropy.best_params_
best_score_entropy = grid_search_entropy.best_score_


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model: {accuracy:.2f}')

# Displaying the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Showing additional metrics
print(classification_report(y_test, y_pred))


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='coolwarm', annot_kws={"size": 10}, fmt='d' )
plt.xlabel('Predicted')
plt.ylabel('Actual')

from sklearn.ensemble import RandomForestClassifier

# Setting up the GridSearchCV object for random forest
grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,  # 5-fold cross validation
    scoring='recall'
)

# Adjusting the GridSearchCV object
grid_search_rf.fit(X_train_resampled, y_train_resampled)
y_pred = grid_search_rf.predict(X_test)

# Obtaining the best parameters and the best scores
best_params_rf = grid_search_rf.best_params_
best_score_rf = grid_search_rf.best_score_

# Displaying the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Showing additional metrics
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='coolwarm', annot_kws={"size": 10}, fmt='d' )
plt.xlabel('Predicted')
plt.ylabel('Actual')

