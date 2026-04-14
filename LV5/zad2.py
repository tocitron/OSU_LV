import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# =========================================================
# IMPORT IZ TEMPLATE-A
# =========================================================
from zadatak_2_template import X_train, X_test, y_train, y_test, plot_decision_regions

# =========================================================
# UCITAVANJE PODATAKA (potrebno za f)
# =========================================================
df = pd.read_csv("penguins.csv")

# ukloni 'sex' i redove s NaN
df = df.drop(columns=['sex'])
df.dropna(axis=0, inplace=True)

# kodiranje klase u integer
species_mapping = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
df['species'] = df['species'].map(species_mapping)

# =========================================================
# SIGURNA KONVERZIJA X_train/X_test u float i y u int
# =========================================================
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# mapiraj y_train i y_test na integer koristeći dictionary
y_train = np.array([species_mapping[s] for s in y_train.ravel()]).astype(int)
y_test  = np.array([species_mapping[s] for s in y_test.ravel()]).astype(int)

# =========================================================
# a) Broj primjera po klasama
# =========================================================
classes, counts_train = np.unique(y_train, return_counts=True)
_, counts_test = np.unique(y_test, return_counts=True)
x = np.arange(len(classes))

plt.figure()
plt.bar(x - 0.2, counts_train, width=0.4, label='Train')
plt.bar(x + 0.2, counts_test, width=0.4, label='Test')
plt.xticks(x, classes)
plt.xlabel('Klasa')
plt.ylabel('Broj primjera')
plt.title('Distribucija klasa')
plt.legend()
plt.show()

# =========================================================
# b) Model logističke regresije (2 feature-a)
# =========================================================
model = LogisticRegression()
model.fit(X_train, y_train)

# =========================================================
# c) Parametri modela
# =========================================================
print("\nIntercepti:", model.intercept_)
print("Koeficijenti:\n", model.coef_)

# =========================================================
# d) Granice odluke
# =========================================================
plot_decision_regions(X_train, y_train, model)
plt.xlabel("bill_length_mm")
plt.ylabel("flipper_length_mm")
plt.title("Granice odluke (train)")
plt.legend()
plt.show()

# =========================================================
# e) Evaluacija na test skupu
# =========================================================
y_pred = model.predict(X_test)

print("\n--- Evaluacija na test skupu (2 feature-a) ---")
print("Matrica zabune:\n", confusion_matrix(y_test, y_pred))
print("Točnost:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# =========================================================
# f) Dodavanje više ulaznih veličina (4 feature-a)
# =========================================================
input_vars_full = ['bill_length_mm', 'flipper_length_mm', 'bill_depth_mm', 'body_mass_g']

# drop NaN i eksplicitna konverzija tipova
df_full = df.dropna(subset=input_vars_full + ['species'])
X_full = df_full[input_vars_full].astype(float).to_numpy()
y_full = df_full['species'].astype(int).to_numpy()

# train/test split
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_full, test_size=0.2, random_state=123
)

# treniranje modela
model_full = LogisticRegression(max_iter=1000)
model_full.fit(X_train_full, y_train_full)

# predikcija i evaluacija
y_pred_full = model_full.predict(X_test_full)
print("\n--- Rezultati sa 4 ulazne veličine ---")
print("Točnost:", accuracy_score(y_test_full, y_pred_full))
print("Matrica zabune:\n", confusion_matrix(y_test_full, y_pred_full))
cm = confusion_matrix(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

classes = ['Adelie', 'Chinstrap', 'Gentoo']

plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = range(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# ispiši brojeve unutar kvadrata
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)

plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.tight_layout()
plt.show()

print("\nClassification report:\n", classification_report(y_test_full, y_pred_full))