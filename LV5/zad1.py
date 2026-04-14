import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# =========================================================
# PRETPOSTAVKA:
# Skripta zadatak_1.py daje sljedeće varijable:
# X_train, y_train, X_test, y_test
# =========================================================

from zadatak_1_template import X_train, y_train, X_test, y_test

# =========================================================
# a) Prikaz podataka
# =========================================================
plt.figure(figsize=(8,6))

# trening skup
plt.scatter(X_train[:,0], X_train[:,1],
            c=y_train, cmap='bwr', label='Train')

# test skup (drugi marker)
plt.scatter(X_test[:,0], X_test[:,1],
            c=y_test, cmap='bwr', marker='x', label='Test')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Podaci (train + test)')
plt.legend()
plt.show()


# =========================================================
# b) Logistic regression model
# =========================================================
model = LogisticRegression()
model.fit(X_train, y_train)


# =========================================================
# c) Parametri i granica odluke
# =========================================================
theta0 = model.intercept_[0]
theta1, theta2 = model.coef_[0]

print("Parametri modela:")
print("theta0 =", theta0)
print("theta1 =", theta1)
print("theta2 =", theta2)

# crtanje granice odluke
x_vals = np.linspace(X_train[:,0].min(), X_train[:,0].max(), 100)
y_vals = -(theta0 + theta1 * x_vals) / theta2

plt.figure(figsize=(8,6))
plt.scatter(X_train[:,0], X_train[:,1],
            c=y_train, cmap='bwr')

plt.plot(x_vals, y_vals, 'k-', label='Granica odluke')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Granica odluke')
plt.legend()
plt.show()


# =========================================================
# d) Evaluacija na testnom skupu
# =========================================================
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print("\nMatrica zabune:")
print(cm)

print("\nTočnost:", acc)
print("Preciznost:", prec)
print("Odziv:", rec)


# =========================================================
# e) Vizualizacija točnih i pogrešnih klasifikacija
# =========================================================
correct = y_pred == y_test

plt.figure(figsize=(8,6))

# točno klasificirani (zeleno)
plt.scatter(X_test[correct, 0], X_test[correct, 1],
            c='green', label='Točno')

# pogrešno klasificirani (crno)
plt.scatter(X_test[~correct, 0], X_test[~correct, 1],
            c='black', label='Pogrešno')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Rezultati klasifikacije (test)')
plt.legend()
plt.show()

