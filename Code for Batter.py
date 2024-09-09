#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')
#load the data
file_path = '/path/batter_raw 2.xlsx'
data = pd.read_excel(file_path)
data.head


# In[ ]:


# Filter out rows where 'BALLS' and 'RUNS' columns are NaN, indicating the player did not bat
data_filtered = data.dropna(subset=['BALLS', 'RUNS'], how='all')

# Define categories based on the OUT TYPE
struck_out_types = ['ct & b', 'c / b', 'b', 'st / b', 'lbw / b', 'hit wicket / b', 'run out']
not_struck_out_types = ['retired out', 'retired not out', 'not out']

# Initialize new columns for OUT and NOT_OUT categories
data_filtered['OUT'] = 0
data_filtered['NOT_OUT'] = 0

# Mark occurrences based on OUT TYPE
data_filtered.loc[data_filtered['OUT TYPE'].isin(struck_out_types), 'OUT'] = 1
data_filtered.loc[data_filtered['OUT TYPE'].isin(not_struck_out_types), 'NOT_OUT'] = 1

# Calculate the average balls per run for entries with valid (non-zero) balls and runs
valid_entries = data_filtered[(data_filtered['BALLS'] > 0) & (data_filtered['RUNS'] >= 0)]
average_balls_per_run = valid_entries['BALLS'].sum() / valid_entries['RUNS'].sum()

# Impute 'BALLS' for entries with zero balls but non-zero runs
data_filtered.loc[(data_filtered['BALLS'] == 0) & (data_filtered['RUNS'] > 0), 'BALLS'] = data_filtered['RUNS'] * average_balls_per_run

data_filtered.head(), data.shape, data_filtered.shape, average_balls_per_run


# In[ ]:


# Recalculate the Strike Rate (SR)
data_filtered['SR'] = (data_filtered['RUNS'] / data_filtered['BALLS']) * 100
data_filtered['SR'] = data_filtered['SR'].round(2)
data_filtered['BALLS'] = data_filtered['BALLS'].round(2)

# Selecting and renaming columns
batter_data = data_filtered[['BATTER','RUNS','BALLS','4s', '6s','SR','OUT','NOT_OUT']]


batter_data.head()


# In[ ]:


# Find the highest score
highest_scores = batter_data.groupby('BATTER')['RUNS'].max().rename('Highest Score')

# Join the highest score back to the original dataframe
batter_data = batter_data.merge(highest_scores, on='BATTER', how='left')

batter_data


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Aggregation for Batters
batter_aggregated = batter_data.groupby('BATTER').agg(
    APPEARANCES=('BATTER', 'size'),
    TOTAL_RUNS=('RUNS', 'sum'),
    TOTAL_BALLS=('BALLS', 'sum'),
    TOTAL_4s=('4s', 'sum'),
    TOTAL_6s=('6s', 'sum'),
    TOTAL_OUT=('OUT', 'sum'),
    HS=('Highest Score','mean'),
    RUNS_AVG=('RUNS', 'mean'),
    SR_AVG=('SR', 'mean'),
    #BR=('BOUNDARY_RATIO', 'mean')
).reset_index()

# Filter out batters 
batter_aggregated = batter_aggregated[(batter_aggregated['TOTAL_RUNS'] > 3) & (batter_aggregated['APPEARANCES'] > 0)]

batter_aggregated['BN_AVG']=(batter_aggregated['TOTAL_4s'] + batter_aggregated['TOTAL_6s'])/batter_aggregated['APPEARANCES']
# probability of getting out
batter_aggregated['PGO'] = (batter_aggregated['TOTAL_OUT'] / batter_aggregated['TOTAL_BALLS'])*100

# Round the columns to two decimal
batter_aggregated['RUNS_AVG'] = batter_aggregated['RUNS_AVG'].round(2)
batter_aggregated['SR'] = batter_aggregated['SR_AVG'].round(2)
batter_aggregated['NB'] = batter_aggregated['BN_AVG'].round(2)
batter_aggregated['PGO'] = batter_aggregated['PGO'].round(2)

# Selecting columns for the final output
batter_aggregated = batter_aggregated[['BATTER','RUNS_AVG','SR','NB', 'HS','PGO']]

batter_aggregated.head(), batter_aggregated.shape


# In[ ]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Select only numeric features for clustering
numeric_data = batter_aggregated.select_dtypes(include=[np.number])

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(numeric_data)

def elbow_method(features_scaled):
    # Calculate the Sum of Squared Distances (SSD) for a range of k values
    ssd = []
    k_values = range(2, 11)  # Testing for 2 to 10 clusters
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        ssd.append(kmeans.inertia_)  # Inertia: Sum of squared distances of samples to their closest cluster center

    # Plot the SSD for each k
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, ssd, 'bo-')
    plt.xlabel('Number of Clusters, k')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    plt.show()

# Call the function to perform elbow method analysis
elbow_method(features_scaled)


# In[ ]:



from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Selecting numeric columns for clustering, if clustering is the next step
features = batter_aggregated.select_dtypes(include=['float64', 'int64'])

# Standardizing the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# KMeans clustering with standardized data
kmeans = KMeans(
    n_clusters=3, 
    random_state=70
)
batter_aggregated['Cluster'] = kmeans.fit_predict(features_scaled)

# Display the first few entries with cluster labels
print(batter_aggregated.head())


# In[ ]:


import scipy.stats as stats
import pandas as pd

metrics = ['RUNS_AVG', 'SR', 'NB', 'HS', 'PGO']  
# Perform ANOVA for each metric
anova_results = {}
for metric in metrics:
    grouped_data = [batter_aggregated[metric][batter_aggregated['Cluster'] == i] for i in range(3)]
    F, p = stats.f_oneway(*grouped_data)
    anova_results[metric] = (F, p)

# Create a DataFrame from the ANOVA results
anova_df = pd.DataFrame.from_dict(anova_results, orient='index', columns=['F-value', 'p-value'])

# Save to CSV
anova_df.to_csv('batter_table2_anova.csv')

# Display ANOVA results
for metric, result in anova_results.items():
    F, p = result
    print(f"ANOVA results for {metric}: F = {F:.3f}, p = {p:.3f}")


# In[ ]:





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
cluster_analysis = batter_aggregated.groupby('Cluster').agg({
    'RUNS_AVG': ['count', 'mean', 'min', 'max'],
    'SR': ['mean', 'min', 'max'],
    'NB': ['mean', 'min', 'max'],
    'HS': ['mean', 'min', 'max'],
    'PGO': ['mean', 'min', 'max']
}).round(3)

# Create multi-level column names
cluster_analysis.columns = ['_'.join(col).upper() for col in cluster_analysis.columns.values]
cluster_analysis = cluster_analysis.stack().unstack(level=0)
#rename 'RUNS_AVG_COUNT' to 'COUNT'
new_levels = [x.replace('RUNS_AVG_COUNT', 'COUNT') for x in cluster_analysis.index]
cluster_analysis.index = new_levels

print("Summary Statistics by Cluster:")
print(cluster_analysis)
# Visualizations for each feature across clusters
features = ['RUNS_AVG', 'SR', 'NB', 'HS', 'PGO']

# Set up the matplotlib figure
plt.figure(figsize=(18, 10))  # Adjust the size of the figure as needed
for i, feature in enumerate(features, 1):
    plt.subplot(2, 3, i)  # Adjust the grid dimensions (e.g., 2 rows, 3 columns) based on the number of features
    sns.boxplot(x='Cluster', y=feature, data=batter_aggregated)
    plt.title(f'Boxplot of {feature}')
    plt.xlabel('Cluster')
    plt.ylabel(feature)

# Adjust layout for better fit and display the plots
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:


#三个聚类标记
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Selecting numeric columns for clustering, if clustering is the next step
features = batter_aggregated.select_dtypes(include=['float64', 'int64'])

# Standardizing the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, max_iter=300)
cluster_labels = kmeans.fit_predict(features_scaled)

# Display the first few entries with cluster labels
print(batter_aggregated.head())


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Encode cluster labels to integers for plotting
label_encoder = LabelEncoder()
encoded_clusters = label_encoder.fit_transform(batter_aggregated['Cluster'])

# Applying PCA to reduce dimensions to 2 for visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])

# Concatenating DataFrame along axis 1 to get final DataFrame before plotting
final_df = pd.concat([principal_df, pd.DataFrame(encoded_clusters, columns=['Encoded Cluster'])], axis=1)

# Customizing plot colors based on the uploaded image style
colors = ['pink', 'c', 'purple']  # Custom colors for each cluster
cluster_labels = ['Cluster 0', 'Cluster 1', 'Cluster 2']  # Labels for each cluster

# Plotting the clusters with customized settings
plt.figure(figsize=(10, 7))
for i, color in enumerate(colors):
    plt.scatter(final_df[final_df['Encoded Cluster'] == i]['Principal Component 1'], 
                final_df[final_df['Encoded Cluster'] == i]['Principal Component 2'], 
                color=color, 
                label=cluster_labels[i])

plt.title('2D PCA of Clustered Data - Batter')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)  # Optional: Adds grid to the plot
plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mapping the cluster numbers to names
cluster_names = {0: 'Better Batter', 1: 'Ordinary Batter', 2: 'Best Batter'}
batter_aggregated['Cluster'] =  batter_aggregated['Cluster'].map(cluster_names)


# Counting the number of players in each tag
tag_counts = batter_aggregated['Cluster'].value_counts()

# Displaying the best bowlers table
from IPython.display import display

best_batters = batter_aggregated[batter_aggregated['Cluster'] == 'Best Batter']
better_batters = batter_aggregated[batter_aggregated['Cluster'] == 'Better Batter']
ordinary_batters = batter_aggregated[batter_aggregated['Cluster'] == 'Ordinary Batter']

# Set display options
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)     
pd.set_option('display.max_colwidth', None) 
pd.set_option('display.width', None)        

print(tag_counts)
print(display(best_batters))
print(best_batters.shape)


# In[ ]:


#Build Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Excluding the player's name and extracting features and labels
features = batter_aggregated.iloc[:, 1:-1]  # Selecting all columns except the first (name) and the last two (cluster and label)
labels = batter_aggregated['Cluster']  # This is the last column which is the label

# Encoding the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.3, random_state=42)

# Showing the shapes of the splits to verify
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


import pandas as pd
from IPython.display import display
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


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

# Display the results of the XGBoost model in a table format
display(cv_df)


# In[ ]:





# In[ ]:


# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
plot_confusion_matrix(xgb_classifier, X_test, y_test)
plt.title('XGBoost Model Confusion Matrix - Batter')
plt.show()


# Feature Importance
xgb.plot_importance(xgb_classifier)
plt.title('Feature Importance')
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

# Preprocessing and SVM pipeline with default parameters
svm_pipeline = make_pipeline(StandardScaler(), SVC(random_state=42))

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Configure the SVM in the pipeline to use the computed class weights
svm_pipeline.set_params(svc__class_weight=class_weight_dict)

# Train the model
svm_pipeline.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = svm_pipeline.predict(X_test)
SVM_report = classification_report(y_test, y_pred, output_dict=True)

# Display the classification report in a DataFrame
SVM_report = pd.DataFrame(SVM_report).transpose()

# Display classification report
print("Classification Report:")
display(SVM_report)


# In[ ]:


# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
plot_confusion_matrix(svm_pipeline, X_test, y_test)
plt.title('SVM Model Confusion Matrix - Batter')
plt.show()


# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


# Initialize RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=90, random_state=42)

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
plt.title('Random Forest Confusion Matrix - Batter')
plt.show()

