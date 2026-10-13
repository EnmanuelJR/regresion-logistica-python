
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from modelo_logistico import entrenar_modelo
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
# Variable para almacenar los resultados del entrenamiento
resultados_modelo = None




# Función para seleccionar el dataset
def cargar_dataset():
    archivo = filedialog.askopenfilename(
        title="Seleccionar dataset de burnout",
        filetypes=[("Archivos CSV", "*.csv")]
    )

    if archivo:
        ruta_dataset.set(archivo)
        etiqueta_archivo.config(
            text="Dataset seleccionado correctamente",
            fg="green"
        )


def ejecutar_entrenamiento():
    global resultados_modelo

    # Verificar que se haya seleccionado un dataset
    if not ruta_dataset.get():
        messagebox.showwarning(
            "Dataset no seleccionado",
            "Primero debes seleccionar un archivo CSV."
        )
        return

    try:
        # Entrenar el modelo con el dataset seleccionado
        resultados_modelo = entrenar_modelo(ruta_dataset.get())
        resultados = resultados_modelo

        # Mostrar las métricas
        etiqueta_resultados.config(
            text=(
                f"Exactitud: {resultados['exactitud']:.2%}\n"
                f"Precisión: {resultados['precision']:.2%}\n"
                f"Sensibilidad: {resultados['sensibilidad']:.2%}"
            )
        )

    except Exception as error:
        messagebox.showerror(
            "Error durante el entrenamiento",
            str(error)
        )


# Función para mostrar la matriz de confusión
def mostrar_matriz_confusion():

    # Verificar que el modelo haya sido entrenado
    if resultados_modelo is None:
        messagebox.showwarning(
            "Modelo no entrenado",
            "Primero debes entrenar el modelo."
        )
        return

    # Crear la visualización
    visualizacion = ConfusionMatrixDisplay(
        confusion_matrix=resultados_modelo["matriz"],
        display_labels=resultados_modelo["categorias"]
    )

    # Mostrar la matriz
    visualizacion.plot(cmap="Blues", values_format="d")

    plt.title("Matriz de confusión - Regresión logística")
    plt.tight_layout()
    plt.show()


# Crear ventana principal
ventana = tk.Tk()

ventana.title("Regresión Logística - Predicción de Burnout")
ventana.geometry("950x650")
ventana.configure(bg="#F4F6F9")

# Variable para almacenar la ruta del dataset
ruta_dataset = tk.StringVar()


# Título principal
titulo = tk.Label(
    ventana,
    text="SISTEMA DE PREDICCIÓN DE BURNOUT",
    font=("Arial", 20, "bold"),
    bg="#F4F6F9",
    fg="#1E3A5F"
)

titulo.pack(pady=25)

# Subtítulo
subtitulo = tk.Label(
    ventana,
    text="Modelo de clasificación mediante regresión logística",
    font=("Arial", 12),
    bg="#F4F6F9",
    fg="#555555"
)

subtitulo.pack(pady=5)


# Sección para cargar el dataset
marco_dataset = tk.Frame(
    ventana,
    bg="#F4F6F9"
)

marco_dataset.pack(pady=35)

# Botón para seleccionar archivo
boton_cargar = tk.Button(
    marco_dataset,
    text="Seleccionar Dataset CSV",
    font=("Arial", 12, "bold"),
    bg="#2563EB",
    fg="white",
    padx=20,
    pady=10,
    command=cargar_dataset
)

boton_cargar.pack(pady=10)

# Etiqueta de estado
etiqueta_archivo = tk.Label(
    marco_dataset,
    text="No se ha seleccionado ningún archivo",
    font=("Arial", 11),
    bg="#F4F6F9",
    fg="#666666"
)

etiqueta_archivo.pack(pady=10)


# Botón para entrenar el modelo
boton_entrenar = tk.Button(
    ventana,
    text="Entrenar modelo",
    font=("Arial", 12, "bold"),
    bg="#16A34A",
    fg="white",
    padx=25,
    pady=10,
    command=ejecutar_entrenamiento
)

boton_entrenar.pack(pady=15)


# Etiqueta para mostrar los resultados del modelo
etiqueta_resultados = tk.Label(
    ventana,
    text="Los resultados aparecerán aquí después del entrenamiento.",
    font=("Arial", 13),
    bg="#F4F6F9",
    fg="#1E3A5F",
    justify="left"
)

etiqueta_resultados.pack(pady=20)


# Botón para visualizar la matriz de confusión
boton_matriz = tk.Button(
    ventana,
    text="Ver matriz de confusión",
    font=("Arial", 12, "bold"),
    bg="#7C3AED",
    fg="white",
    padx=20,
    pady=10,
    command=mostrar_matriz_confusion
)

boton_matriz.pack(pady=15)


# Iniciar aplicación
ventana.mainloop()
