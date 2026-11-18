
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load handwritten digit dataset
digits = load_digits(n_class=10)

X = digits.data
y = digits.target

# 2. Display a sample digit
plt.imshow(digits.images[0], cmap='gray')
plt.title("Digit: " + str(y[0]))
plt.show()

# 3. Split data into training and testing
X_train = X[:1500]
y_train = y[:1500]

X_test = X[1501:]
y_test = y[1501:]

# 4. Create and train SVM model
model = svm.SVC(gamma=0.001, C=100)
model.fit(X_train, y_train)

# 5. Predict digits
predictions = model.predict(X_test)

print("Predicted:", predictions)
print("Actual:", y_test)

# 6. Confusion Matrix
cm = confusion_matrix(y_test, predictions)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 7. Classification Report
print(classification_report(y_test, predictions))
