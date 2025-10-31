import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def cargar_grafo(ruta_archivo):
    # Leer el CSV
    df = pd.read_csv(ruta_archivo)
    df.columns = df.columns.str.strip()  # limpia espacios
    # Crear grafo
    G = nx.DiGraph()

    # Recorre cada fila
    for _, fila in df.iterrows():
        padre = str(fila["Padre"]).strip()
        # Ignorar filas vacías
        if padre.lower() == "nan" or padre == "":
            continue
        # Iterar sobre las columnas que empiecen por 'Hijo'
        for col in df.columns:
            if "Hijo" in col:
                hijo = str(fila[col]).strip()
                if hijo.lower() != "nan" and hijo != "":
                    G.add_edge(padre, hijo)
    return G

def mostrar_grafo(grafo, titulo="Red Bayesiana - Dependencias"):
    """
    Dibuja en pantalla el grafo de la red bayesiana con formato visual claro.
    Parámetros:
        grafo: objeto networkx.DiGraph() con las relaciones cargadas.
        titulo: texto del título mostrado sobre el gráfico.
    """
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(grafo, seed=42)
    nx.draw(
        grafo, pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2500,
        arrowsize=20,
        font_size=10,
        font_weight="bold"
    )
    plt.title(titulo)
    plt.show()
    
# === Crear y dibujar el grafo ===
grafo = cargar_grafo(r"C:\Users\bibliotecapuj\Desktop\IA\proyecto_3_IA\Dependencias.csv")
mostrar_grafo(grafo)