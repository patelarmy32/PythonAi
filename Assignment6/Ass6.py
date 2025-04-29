import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('bank.csv', delimiter=';')

print(df.info())
print(df.head())

df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]
print(df2.head())

df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])


df3['y'] = df3['y'].apply(lambda x: 1 if x == 'yes' else 0)

print(df3.head())

plt.figure(figsize=(16, 12))
sns.heatmap(df3.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.show()

X = df3.drop('y', axis=1)
y = df3['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

cm_logreg = confusion_matrix(y_test, y_pred_logreg)
acc_logreg = accuracy_score(y_test, y_pred_logreg)

print("Confusion Matrix (Logistic Regression):\n", cm_logreg)
print("Accuracy (Logistic Regression): {:.2f}".format(acc_logreg))

plt.figure(figsize=(6, 4))
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)

print("Confusion Matrix (KNN, k=3):\n", cm_knn)
print("Accuracy (KNN, k=3): {:.2f}".format(acc_knn))

plt.figure(figsize=(6, 4))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix: KNN (k=3)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

