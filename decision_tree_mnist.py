# -------------------------------------------------------
# Árboles de Decisión - Comparación de Profundidades
# Dataset: MNIST (flattened)
# Compatible con PyCharm
# -------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# -------------------------------------------------------
# 2️⃣ Carga y exploración del dataset
# -------------------------------------------------------

print("Cargando dataset...")

df = pd.read_csv(r"C:\Temp-univ\Asig-3\mnist_train.csv")

print("\nPrimeras filas del dataset:")
print(df.head())

print("\nDimensiones del dataset:")
print(f"Filas: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")

print("\nVerificando valores nulos:")
print("Total valores nulos:", df.isnull().sum().sum())

print("\nComentario:")
print("Cada fila representa una imagen MNIST aplanada (28x28 pixeles).")
print("Las columnas contienen los valores de los pixeles y la etiqueta del dígito.")

# -------------------------------------------------------
# 3️⃣ Preparación de X e y
# -------------------------------------------------------

X = df.iloc[:, :-1]  # 784 pixeles
y = df.iloc[:, -1]  # etiqueta

print("\nClases presentes en y:", np.unique(y))
print("Cantidad de clases:", y.nunique())

print("\nComentario:")
print("X contiene las imágenes aplanadas.")
print("y representa el dígito real (0 a 9).")

# -------------------------------------------------------
# 4️⃣ División en entrenamiento y prueba
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nCantidad de ejemplos:")
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))

# -------------------------------------------------------
# 5️⃣ Profundidades a evaluar
# -------------------------------------------------------

profundidades = [5, 10, 20]

print("\nProfundidades evaluadas:", profundidades)
print("Comparar varias profundidades permite analizar subajuste y sobreajuste")

# -------------------------------------------------------
# 6️⃣ Entrenamiento de modelos
# -------------------------------------------------------

resultados = []

print("\nEntrenando modelos...\n")

for profundidad in profundidades:
    print(f"Entrenando árbol con profundidad = {profundidad}")

    modelo = DecisionTreeClassifier(
        max_depth=profundidad,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    resultados.append([profundidad, acc_train, acc_test])

    print(f"Accuracy Train: {acc_train:.4f}")
    print(f"Accuracy Test : {acc_test:.4f}\n")

# -------------------------------------------------------
# 7️⃣ Tabla de comparación
# -------------------------------------------------------

df_resultados = pd.DataFrame(
    resultados,
    columns=["Profundidad", "Accuracy Train", "Accuracy Test"]
)

print("\nTabla de resultados:")
print(df_resultados)

print("\nAnálisis:")
print("La profundidad con mejor accuracy en prueba suele ser intermedia.")
print("Una gran diferencia entre train y test indica sobreajuste.")

# -------------------------------------------------------
# 8️⃣ Gráfica de desempeño
# -------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(df_resultados["Profundidad"], df_resultados["Accuracy Train"],
         marker='o', label="Entrenamiento")
plt.plot(df_resultados["Profundidad"], df_resultados["Accuracy Test"],
         marker='o', label="Prueba")

plt.xlabel("Profundidad del Árbol")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Profundidad del Árbol")
plt.legend()
plt.grid(True)
plt.show()

print("\nInterpretación:")
print("Aumentar la profundidad no siempre mejora el modelo.")
print("El sobreajuste ocurre cuando el accuracy de entrenamiento aumenta")
print("pero el de prueba se estanca o disminuye.")

# -------------------------------------------------------
# 9️⃣ Visualización del árbol (profundidad baja)
# -------------------------------------------------------

print("\nVisualizando árbol con profundidad 5...")

modelo_visual = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo_visual.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(
    modelo_visual,
    max_depth=2,
    filled=True,
    fontsize=8
)
plt.show()

print("\nComentario:")
print("El árbol toma decisiones basadas en pixeles individuales.")
print("Aunque interpretable, para MNIST resulta poco intuitivo.")

# -------------------------------------------------------
# 🔟 Conclusiones finales
# -------------------------------------------------------

print("\nCONCLUSIONES FINALES")
print("- Los árboles de decisión funcionan razonablemente en MNIST.")
print("- Profundidades intermedias ofrecen mejor balance.")
print("- Profundidades bajas sufren subajuste.")
print("- Profundidades altas generan sobreajuste.")
print("- Evaluar distintos modelos es clave para una buena generalización.")