import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# Load the data
data = pd.read_csv(r"C:\Users\souja\Downloads\archive (2)\mushrooms.csv")

# Data cleaning
print("Data Cleaning:")
print(f"Initial shape of the data: {data.shape}")
data.drop_duplicates(inplace=True)
print(f"Shape after removing duplicates: {data.shape}")

# Handle missing values (if any)
if data.isnull().sum().sum() > 0:
    data.fillna(data.mode().iloc[0], inplace=True)
    print("Filled missing values with mode.")

# Data integration (assuming there's more data to integrate, but here it's just a single dataset)
# Normally, you would merge or join different datasets here
print("Data Integration: No additional datasets to integrate.")

# Data transformation
print("Data Transformation:")
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])
    print(f"Encoded {column}")

# Feature scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print("Data standardized using StandardScaler.")

# Feature and target separation
X = data.drop('class', axis=1)
y = data['class']
print("Features and target variable separated.")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data split into training and testing sets.")

# Select only two features for SVM visualization
X_train_svm = X_train_scaled[:, :2]
X_test_svm = X_test_scaled[:, :2]

# Function to print results and plot confusion matrix
def print_results_and_plot(model_name, y_test, y_pred, model=None):
    print(f"\n{model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Plot decision tree if model is Decision Tree or Random Forest
    if model_name in ["Decision Tree", "Random Forest"]:
        plt.figure(figsize=(20, 10))
        if model_name == "Random Forest":
            estimator = model.estimators_[0]
        else:
            estimator = model
        plot_tree(estimator, filled=True, feature_names=X.columns, class_names=le.classes_)
        plt.title(f"{model_name} - Tree Plot")
        plt.show()

    # Plot linear regression line with actual and predicted points connected
    if model_name == "Linear Regression":
        plt.figure(figsize=(6, 4))
        plt.scatter(X_test_scaled[:, 0], y_test, color='blue', label='Actual')
        plt.scatter(X_test_scaled[:, 0], y_pred, color='red', marker='o', label='Predicted')
        plt.title(f"{model_name}")
        plt.xlabel('Feature 1 (Standardized)')
        plt.ylabel('Class')
        plt.legend()
        plt.show()

    # Plot logistic regression sigmoid curve
    if model_name == "Logistic Regression":
        plt.figure(figsize=(6, 4))
        plt.scatter(X_test_scaled[:, 0], y_test, color='blue', label='Actual')
        X_test_sorted = np.sort(X_test_scaled[:, 0])
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        plt.plot(X_test_sorted, 1 / (1 + np.exp(-X_test_sorted * model.coef_[0][0] - model.intercept_[0])), color='red', label='Sigmoid Curve')
        plt.title(f"{model_name}")
        plt.xlabel('Feature 1 (Standardized)')
        plt.ylabel('Probability')
        plt.legend()
        plt.show()

    # Plot SVM decision boundary for non-linear SVM
    if model_name == "Support Vector Machine":
        plt.figure(figsize=(6, 4))
        X_set, y_set = X_test_svm, y_test
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
        plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha=0.75, cmap=plt.cm.Paired)
        plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, edgecolor='k', s=20, cmap=plt.cm.Paired)
        plt.title(f"{model_name}")
        plt.xlabel('Feature 1 (Standardized)')
        plt.ylabel('Feature 2 (Standardized)')
        plt.show()

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled).round()
print_results_and_plot("Linear Regression", y_test, y_pred_lr)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)
print_results_and_plot("Logistic Regression", y_test, y_pred_log_reg, model=log_reg)

# SVM
svm = SVC(kernel='rbf')
svm.fit(X_train_svm, y_train)
y_pred_svm = svm.predict(X_test_svm)
print_results_and_plot("Support Vector Machine", y_test, y_pred_svm, model=svm)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)
print_results_and_plot("Decision Tree", y_test, y_pred_dt, model=dt)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
print_results_and_plot("Random Forest", y_test, y_pred_rf, model=rf)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
print_results_and_plot("Naive Bayes", y_test, y_pred_nb)

# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
print_results_and_plot("K-Nearest Neighbors", y_test, y_pred_knn)

# Visualization of model comparison
models = ["Linear Regression", "Logistic Regression", "Support Vector Machine", "Decision Tree", "Random Forest", "Naive Bayes", "K-Nearest Neighbors"]
accuracies = [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_log_reg), accuracy_score(y_test, y_pred_svm),
              accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_nb),
              accuracy_score(y_test, y_pred_knn)]

plt.figure(figsize=(12, 6))
sns.barplot(x=models, y=accuracies)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.xticks(rotation=45)
plt.show()

# PCA for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_scaled)
plt.figure(figsize=(12, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='viridis', edgecolor='k', s=150)
plt.title('PCA of Mushroom Dataset')
plt.xlabel('Principal Component 1 - Explains most variance')
plt.ylabel('Principal Component 2 - Explains second most variance')
plt.show()
