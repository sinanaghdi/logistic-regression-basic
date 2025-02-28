import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# بارگیری داده‌ها
st.title("داشبورد تحلیل دسته‌بندی با رگرسیون لجستیک")
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# training model
model = LogisticRegression(multi_class='ovr')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# results
st.write("## ماتریس سردرگمی")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

st.write("## گزارش دسته‌بندی")
st.text(classification_report(y_test, y_pred))