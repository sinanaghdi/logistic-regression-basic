import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
st.title("Interactive Logistic Regression Dashboard with Regularization")
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Sidebar for user inputs
st.sidebar.header("Model Parameters")
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)  # User can adjust test size
C = st.sidebar.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.01)  # C = 1/λ
max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 100, 50)  # User can adjust max iterations

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train the model with regularization
model = LogisticRegression(C=C, max_iter=max_iter, penalty='l2')  # L2 Regularization
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display results
st.write("## Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

st.write("## Classification Report")
st.text(classification_report(y_test, y_pred))

# Display feature importance (coefficients)
st.write("## Feature Importance (Coefficients)")
coef_df = pd.DataFrame(model.coef_, columns=data.feature_names, index=data.target_names)
st.dataframe(coef_df)

# Plot feature importance
st.write("## Feature Importance Plot")
fig, ax = plt.subplots()
sns.barplot(x=coef_df.columns, y=coef_df.iloc[0], ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Display raw data
st.write("## Raw Data")
if st.checkbox("Show raw data"):
    st.write(X)