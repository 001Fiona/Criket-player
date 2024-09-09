#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Feature Engineering_Bowler

import pandas as pd
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_excel("/path/bowler_raw.xlsx")
df.head


# In[ ]:


# Calculate balls bowled
df['BALLS_BOWLED'] = df['OVERS'].apply(lambda x: int(x) * 6 + int((x - int(x)) * 10))

# Calculating Dot Ball Percentage
df['TOTAL_DOT_BALLS'] = df['MAIDENS'] * 6
df['PDB'] = (100*df['TOTAL_DOT_BALLS'] / df['BALLS_BOWLED'])

# Feature Engineering for Number of Wides and No Balls
df['NWB'] = df['WIDES'] + df['NO BALLS']

# Aggregate data to calculate totals and averages, including appearances
df_aggregated = df.groupby('BOWLER').agg(
    TOTAL_WICKETS=('WICKETS', 'sum'),
    TOTAL_RUNS=('RUNS', 'sum'),
    TOTAL_BALLS=('BALLS_BOWLED','sum'),
    APPEARANCES=('BOWLER', 'size'),
    AVG_PDB=('PDB', 'mean'),
    AVG_NWB=('NWB', 'mean'),
    AVG_ECON=('ECON', 'mean')
)

# Filter out bowlers 
df_aggregated = df_aggregated[(df_aggregated['APPEARANCES'] > 1) & (df_aggregated['TOTAL_WICKETS'] > 0)]

# Calculate feature
df_aggregated['BAV'] = (df_aggregated['TOTAL_RUNS'] / df_aggregated['TOTAL_WICKETS'])
df_aggregated['SRB'] = (df_aggregated['TOTAL_BALLS'] / df_aggregated['TOTAL_WICKETS'])

# Prepare the final DataFrame with specified columns
final_df = df_aggregated.reset_index()
final_df['ECON_AVG'] = final_df['AVG_ECON'].round(3)
final_df['WICKETS_AVG'] = (final_df['TOTAL_WICKETS'] / final_df['APPEARANCES']).round(3)
final_df['RUNS_AVG'] = (final_df['TOTAL_RUNS'] / final_df['APPEARANCES']).round(3)
final_df['BALLS_AVG'] = (final_df['TOTAL_BALLS'] / final_df['APPEARANCES']).round(3)
final_df['BAV'] = final_df['BAV'].round(3)
final_df['SRB'] = final_df['SRB'].round(3)
final_df['NWB'] = final_df['AVG_NWB'].round(3)
final_df['PDB'] = final_df['AVG_PDB'].round(3)

# Selecting and renaming columns for the final output
final_output = final_df[['BOWLER','ECON_AVG','SRB','BAV', 'NWB','PDB']]

print(final_output.head())
final_output.shape


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Select only numeric features for clustering
numeric_data = final_output.select_dtypes(include=[np.number])

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(numeric_data)

def elbow_method(features_scaled):
    # Calculate the Within-Cluster Sum of Squared Distances (WSS) for a range of k values
    wss = []
    k_values = range(1, 11)  # Typically, the elbow method starts at k=1 to see the decline
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        wss.append(kmeans.inertia_)  # inertia_ is the WSS

    # Plot the WSS values to find the elbow
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, wss, 'bo-')
    plt.xlabel('Number of Clusters, k')
    plt.ylabel('WSS (Inertia)')
    plt.title('Elbow Method For Determining Optimal k')
    plt.show()

# Call the function to perform the elbow method analysis
elbow_method(features_scaled)


# In[ ]:


# K-Means clustering

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Selecting numeric columns for clustering, if clustering is the next step
features = final_output.select_dtypes(include=['float64', 'int64'])

# Standardizing the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
final_output['Cluster'] = kmeans.fit_predict(features_scaled)

# Display the first few entries with cluster labels
print(final_output.head())


# In[ ]:


import scipy.stats as stats

metrics = ['ECON_AVG','SRB','BAV', 'NWB','PDB']  
# Perform ANOVA for each metric
anova_results = {}
for metric in metrics:
    grouped_data = [final_output[metric][final_output['Cluster'] == i] for i in range(3)]
    F, p = stats.f_oneway(*grouped_data)
    anova_results[metric] = (F, p)

# Create a DataFrame from the ANOVA results
anova_df = pd.DataFrame.from_dict(anova_results, orient='index', columns=['F-value', 'p-value'])

# Display ANOVA results
for metric, result in anova_results.items():
    F, p = result
    print(f"ANOVA results for {metric}: F = {F:.3f}, p = {p:.3f}")


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Set display options
pd.set_option('display.max_columns', None)  # Ensures all columns are shown
pd.set_option('display.max_rows', None)     # Ensures all rows are shown
pd.set_option('display.max_colwidth', None) # Ensures full content of each cell is shown
pd.set_option('display.width', None)        # Automatically adjusts the display width to show each line


# Compute summary statistics for each cluster
cluster_analysis = final_output.groupby('Cluster').agg({
    'ECON_AVG': ['count', 'mean', 'min', 'max'],
    'SRB': ['mean', 'min', 'max'],
    'BAV': ['mean', 'min', 'max'],
    'NWB': ['mean', 'min', 'max'],
    'PDB': ['mean', 'min', 'max']
}).round(3)

# Create multi-level column names
cluster_analysis.columns = ['_'.join(col).upper() for col in cluster_analysis.columns.values]
cluster_analysis = cluster_analysis.stack().unstack(level=0)
#rename to 'COUNT'
new_levels = [x.replace('ECON_COUNT', 'COUNT') for x in cluster_analysis.index]
cluster_analysis.index = new_levels

print("Summary Statistics by Cluster:")
print(cluster_analysis)
# Visualizations for each feature across clusters
features = ['ECON_AVG','SRB','BAV', 'NWB','PDB']

# Set up the matplotlib figure
plt.figure(figsize=(18, 10))  # Adjust the size of the figure as needed
for i, feature in enumerate(features, 1):
    plt.subplot(2, 3, i)  # Adjust the grid dimensions (e.g., 2 rows, 3 columns) based on the number of features
    sns.boxplot(x='Cluster', y=feature, data=final_output)
    plt.title(f'Boxplot of {feature}')
    plt.xlabel('Cluster')
    plt.ylabel(feature)

# Adjust layout for better fit and display the plots
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:


#Visualization of the output

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce the dimensions for visualization
pca = PCA(n_components=2)  # Reduce to 2 dimensions for plotting
principal_components = pca.fit_transform(features_scaled)

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Cluster'] = final_output['Cluster']

# Plotting the clusters
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b', 'y', 'c', 'm']
for cluster in pca_df['Cluster'].unique():
    cluster_data = pca_df[pca_df['Cluster'] == cluster]
    plt.scatter(cluster_data['Principal Component 1'], cluster_data['Principal Component 2'], s=50, alpha=0.5, label=f'Cluster {cluster}', color=colors[cluster])
plt.title('2D PCA of Clustered Data - Bowler')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Mapping the cluster numbers to names directly in the DataFrame
cluster_names = {1: 'Better Bowler', 0: 'Ordinary Bowler', 2: 'Best Bowler'}
final_output['Cluster'] = final_output['Cluster'].map(cluster_names)

# Counting the number of players in each tag
tag_counts = final_output['Cluster'].value_counts()

# Filtering to display the best bowlers table
best_bowlers = final_output[final_output['Cluster'] == 'Best Bowler']
better_bowlers = final_output[final_output['Cluster'] == 'Better Bowler']
ordinary_bowlers = final_output[final_output['Cluster'] == 'Ordinary Bowler']

# Set display options for better visibility in output
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)     
pd.set_option('display.max_colwidth', None) 
pd.set_option('display.width', None)        

# Output the count of players in each cluster
print(tag_counts)

# Display the data frame of the best bowlers
display(best_bowlers)

# Print the shape of the best bowlers DataFrame
print(best_bowlers.shape)


# In[ ]:


#Build Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Excluding the player's name and extracting features and labels
features = final_output.iloc[:, 1:-1]  # Selecting all columns except the first (name) and the last two (cluster and label)
labels = final_output['Cluster']  # This is the last column which is the label

# Encoding the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.3, random_state=42)

# Showing the shapes of the splits to verify
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


#XGBoost classifier
from IPython.display import display
from sklearn.model_selection import cross_validate
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, recall_score, accuracy_score, precision_score, f1_score

# Define the XGBoost classifier with default parameters
xgb_classifier = xgb.XGBClassifier()


# Train the model
xgb_classifier.fit(X_train, y_train)

# Defining multiple scoring metrics for evaluation
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': 'precision_macro',
    'recall': 'recall_macro',
    'f1': 'f1_macro'
}

# Cross-validation using multiple metrics
cv_results = cross_validate(xgb_classifier, features, encoded_labels, cv=5, scoring=scoring)

# Building a DataFrame from the cross-validation results
cv_metrics = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
    'Scores': [
        cv_results['test_accuracy'],
        cv_results['test_precision'],
        cv_results['test_recall'],
        cv_results['test_f1']
    ],
    'Mean Score': [
        cv_results['test_accuracy'].mean(),
        cv_results['test_precision'].mean(),
        cv_results['test_recall'].mean(),
        cv_results['test_f1'].mean()
    ]
}

cv_df = pd.DataFrame(cv_metrics)

# Adding individual scores for each fold as separate columns
fold_names = [f'Fold {i+1}' for i in range(5)]
for i, fold_name in enumerate(fold_names):
    cv_df[fold_name] = cv_df['Scores'].apply(lambda x: x[i])

# Drop the 'Scores' list to clean up the table for display
cv_df.drop(columns=['Scores'], inplace=True)

# Display the results of XGBoot model in a table format
display(cv_df)


# In[ ]:


# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
plot_confusion_matrix(xgb_classifier, X_test, y_test)
plt.title('XGBoost Model Confusion Matrix-Bowler')
plt.show()


# Feature Importance
xgb.plot_importance(xgb_classifier)
plt.title('Feature Importance')
plt.show()


# In[ ]:





# In[ ]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Preprocessing and SVM pipeline
svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale', random_state=42))

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Parameter grid for Grid Search
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 0.1, 0.01],
    'svc__kernel': ['rbf', 'linear', 'poly'],
    'svc__class_weight': [class_weight_dict]
}

# Scoring methods
scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Grid search with cross-validation for multiple metrics
grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring=scoring_metrics, refit='accuracy')
grid_search.fit(X_train, y_train)

# Output the best parameters, best score, and mean cross-validation accuracy
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Evaluate the best model on the test set using additional metrics
y_pred = grid_search.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

# Display the classification report in a DataFrame
SVM_report = pd.DataFrame(report).transpose()

# Display classification report
print("Classification Report:")
display(SVM_report)

# Save output to CSV
SVM_report.to_csv("bowler_table7_SVM.csv")


# In[ ]:





# In[ ]:


# Plotting the confusion matrix
plot_confusion_matrix(grid_search, X_test, y_test)
plt.title('SVM Model Confusion Matrix-bowler')
plt.show()


# In[ ]:





# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
# Initialize RandomForestClassifier
rf_classifier = RandomForestClassifier(
    n_estimators=90, 
    #max_depth=3, 
    random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = rf_classifier.predict(X_test)

# Collecting evaluation metrics
report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)

# Converting the classification report into a DataFrame
RF_report = pd.DataFrame(report).transpose()

# Adding accuracy score to the DataFrame
RF_report.loc['accuracy', 'precision'] = accuracy  
# Display the classification report
print("Classification Report:")
display(RF_report)


# In[ ]:



# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
plot_confusion_matrix(rf_classifier, X_test, y_test)
plt.title('Random Forest Confusion Matrix-Bowler')
plt.show()

