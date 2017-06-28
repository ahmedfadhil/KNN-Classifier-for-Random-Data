import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('KNN_Project_Data')
df.head()
# sns.pairplot(df, hue='TARGET CLASS', palette='coolwarm')
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scale_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
df_feat = pd.DataFrame(scale_features, columns=df.columns[:-1])
df_feat.head()

# Train and test split
X = df_feat
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# Decrease error rate to get better k value
error_rate = []
for i in range(1, 70):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10, 4))
plt.plot(range(1, 70), error_rate, color='blue', linestyle='--', marker='o', markerfacecolor='red', markersize=10)

plt.title('Error rate vs k')
plt.xlabel('K')
plt.xlabel("Error Rate")

#  Choosing the new k value

knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test, pred))

print('\n')
print(classification_report(y_test, pred))
