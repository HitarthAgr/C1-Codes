
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
cols = ['buying', 'maint', 'doors', 'persons',
        'lug_boot', 'safety', 'class']

data = pd.read_csv("car.csv", names=cols)

# 2. Encode categorical values into numbers
le = LabelEncoder()

for col in data.columns:
    data[col] = le.fit_transform(data[col])

# 3. Separate input and target
X = data.drop('class', axis=1)
y = data['class']

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Create and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Predict output
y_pred = model.predict(X_test)

# 7. Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 9. Feature Importance
print("Feature Importance:")
print(model.feature_importances_)
