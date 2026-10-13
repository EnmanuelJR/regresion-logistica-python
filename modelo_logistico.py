
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# Función principal del modelo de regresión logística
def entrenar_modelo(ruta_archivo):

    # 1. Cargar el dataset
    df = pd.read_csv(ruta_archivo)

    # 2. Definir la variable objetivo
    y = df["Burnout_Risk"]

    # Convertir las categorías a números
    codificador = LabelEncoder()
    y = codificador.fit_transform(y)

    # 3. Seleccionar variables predictoras
    columnas_excluir = [
        "Person_ID",
        "Burnout_Risk",
        "Burnout_Score",
        "Mental_Health_Status",
        "AI_Wellness_Recommendation"
    ]

    X = df.drop(columns=columnas_excluir)

    # 4. Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    
    # 5. Identificar variables numéricas y categóricas
    variables_numericas = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    variables_categoricas = X.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    # 6. Preparar variables numéricas
    procesador_numerico = Pipeline(steps=[
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler())
    ])

    # 7. Preparar variables categóricas
    procesador_categorico = Pipeline(steps=[
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("codificador", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 8. Unir los procesamientos
    preprocesador = ColumnTransformer(
        transformers=[
            ("numericas", procesador_numerico, variables_numericas),
            ("categoricas", procesador_categorico, variables_categoricas)
        ]
    )

    
    # 9. Crear el modelo de regresión logística
    modelo = Pipeline(steps=[
        ("preprocesamiento", preprocesador),
        ("regresion_logistica", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    # 10. Entrenar el modelo
    modelo.fit(X_train, y_train)

    # 11. Realizar predicciones
    y_pred = modelo.predict(X_test)

    # Convertir predicciones numéricas a categorías
    predicciones = codificador.inverse_transform(y_pred)

    
    # 12. Calcular la exactitud
    exactitud = accuracy_score(y_test, y_pred)

    # 13. Calcular la precisión
    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # 14. Calcular la sensibilidad
    sensibilidad = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # 15. Calcular la matriz de confusión
    matriz = confusion_matrix(y_test, y_pred)

    # 16. Devolver los resultados a la aplicación
    return {
        "exactitud": exactitud,
        "precision": precision,
        "sensibilidad": sensibilidad,
        "matriz": matriz,
        "categorias": codificador.classes_,
        "predicciones": predicciones
    }



