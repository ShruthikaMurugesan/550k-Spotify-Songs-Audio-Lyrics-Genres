import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("/content/artists.csv")

df['main_genre'] = df['main_genre'].fillna('unknown')
df['genres'] = df['genres'].fillna('[]')

df['genre_count'] = df['genres'].apply(lambda x: len(x.split(',')))

def popularity_level(score):
    if score <= 30:
        return 'Low'
    elif score <= 60:
        return 'Medium'
    else:
        return 'High'

df['popularity_level'] = df['popularity'].apply(popularity_level)

sns.countplot(x='popularity_level', data=df)
plt.title("Popularity Level Distribution")
plt.show()

plt.scatter(df['followers'], df['popularity'], alpha=0.3)
plt.xlabel("Followers")
plt.ylabel("Popularity")
plt.title("Followers vs Popularity")
plt.show()

le = LabelEncoder()
df['main_genre_encoded'] = le.fit_transform(df['main_genre'])

X = df[['followers', 'genre_count', 'main_genre_encoded']]
y = df['popularity_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

feature_importance = model.feature_importances_
features = X.columns

plt.barh(features, feature_importance)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance")
plt.show()
