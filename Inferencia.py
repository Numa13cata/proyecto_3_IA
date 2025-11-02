import pandas as pd
import os
import networkx as nx
from RedBayesiana import cargar_grafo
import datetime

def cargar_cpts(carpeta):
    cpts = {}
    # Recorre todos los archivos de la carpeta
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".csv"):
            ruta = os.path.join(carpeta, archivo)

            # Leer el archivo
            df = pd.read_csv(ruta)
            df.columns = df.columns.str.strip()  # limpia espacios

            # El nombre de la variable será el nombre del archivo sin .csv
            nombre_variable = os.path.splitext(archivo)[0]

            # Guarda el DataFrame en el diccionario
            cpts[nombre_variable] = df

    return cpts

def generar_dominios(cpts):
    """
    Genera automáticamente los dominios (valores posibles) de cada variable
    a partir de las tablas de probabilidad condicional (CPTs).
    """
    dominios = {}

    for nombre_var, tabla in cpts.items():
        # La primera columna normalmente es la variable principal
        var_principal = tabla.columns[0]
        
        # Extrae los valores únicos posibles de esa variable
        valores = tabla[var_principal].dropna().unique().tolist()
        
        # Guarda en el diccionario
        dominios[var_principal] = valores

    return dominios

def definir_consulta_y_evidencia():
    """
    Paso 1: Definir la variable de consulta y la evidencia conocida.
    """
    # Solicitar la variable de consulta
    variable_objetivo = input("Ingrese la variable de consulta (por ejemplo, Appointment): ").strip()
    evidencia = {}
    print("\nIngrese la evidencia conocida (por ejemplo, Rain=heavy, Maintenance=yes).")
    print("Cuando termine, presione Enter sin escribir nada.\n")

    # Permitir ingresar pares variable=valor
    while True:
        entrada = input("Evidencia (o Enter para terminar): ").strip()
        if entrada == "":
            break
        try:
            var, val = entrada.split("=")
            var = var.strip()
            val = val.strip()

            # Convertir valores tipo texto a booleano si aplica
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False

            evidencia[var] = val
        except ValueError:
            print("Formato incorrecto. Use la forma Variable=Valor.")

    print("\nConsulta definida:")
    print(f"Variable objetivo: {variable_objetivo}")
    print(f"Evidencia: {evidencia}\n")

    return variable_objetivo, evidencia

def obtener_orden_topologico(grafo):
    """
    Paso 2: Obtener un orden topológico de las variables
    (los padres antes que los hijos).
    """
    return list(nx.topological_sort(grafo))

def inferencia_por_enumeracion(variable, evidencia, grafo, cpts, dominios):
    """
    Ejecuta la inferencia por enumeración y genera una traza paso a paso.
    """
    # Limpiar la traza anterior
    open("traza_inferencia.txt", "w").close()

    registrar_traza(f"\n=== INICIO DE LA INFERENCIA POR ENUMERACIÓN ===")
    registrar_traza(f"Variable objetivo: {variable}")
    registrar_traza(f"Evidencia inicial: {evidencia}\n")

    orden = list(nx.topological_sort(grafo))
    resultados = {}

    # Para cada valor posible de la variable objetivo
    for valor in dominios[variable]:
        e_extendida = evidencia.copy()
        e_extendida[variable] = valor
        registrar_traza(f"\n→ Evaluando hipótesis: {variable} = {valor}")
        prob = enumerar_todo(orden, e_extendida, grafo, cpts, dominios, nivel=1)
        registrar_traza(f"Resultado parcial para {variable}={valor}: {prob:.6f}\n")
        resultados[valor] = prob

    # Normalizar resultados
    total = sum(resultados.values())
    for k in resultados:
        resultados[k] /= total

    registrar_traza(f"=== RESULTADOS NORMALIZADOS ===")
    for val, prob in resultados.items():
        registrar_traza(f"P({variable}={val} | evidencia) = {prob:.4f}")

    registrar_traza(f"\n=== FIN DE LA INFERENCIA ===\n")
    return resultados


def enumerar_todo(variables, evidencia, grafo, cpts, dominios, nivel=0):
    # Caso base: sin variables, retornamos 1
    if not variables:
        return 1.0

    Y = variables[0]
    resto = variables[1:]
    padres = list(grafo.predecessors(Y))
    indent = "   " * nivel  # sangría visual

    # Si la variable tiene un valor conocido (evidencia)
    if Y in evidencia:
        prob = obtener_probabilidad(Y, evidencia[Y], evidencia, cpts, padres)
        if padres:
            registrar_traza(f"{indent}Usando evidencia: P({Y}={evidencia[Y]} | {', '.join([f'{p}={evidencia[p]}' for p in padres])}) = {prob}")
        else:
            registrar_traza(f"{indent}Usando evidencia: P({Y}={evidencia[Y]}) = {prob}")
        return prob * enumerar_todo(resto, evidencia, grafo, cpts, dominios, nivel + 1)

    # Si la variable NO está en la evidencia (hay que sumar sobre sus valores posibles)
    else:
        total = 0
        registrar_traza(f"{indent}Enumerando posibles valores de {Y}: {dominios[Y]}")
        for valor in dominios[Y]:
            nueva_evidencia = evidencia.copy()
            nueva_evidencia[Y] = valor
            prob = obtener_probabilidad(Y, valor, nueva_evidencia, cpts, padres)
            if padres:
                registrar_traza(f"{indent}   Caso {Y}={valor}: P({Y}={valor} | {', '.join([f'{p}={nueva_evidencia[p]}' for p in padres])}) = {prob}")
            else:
                registrar_traza(f"{indent}   Caso {Y}={valor}: P({Y}={valor}) = {prob}")

            subtotal = prob * enumerar_todo(resto, nueva_evidencia, grafo, cpts, dominios, nivel + 1)
            registrar_traza(f"{indent}   Resultado parcial para {Y}={valor}: {subtotal:.5f}")
            total += subtotal

        registrar_traza(f"{indent}Suma total de {Y} = {total:.5f}\n")
        return total


    
def obtener_probabilidad(var, valor, evidencia, cpts, padres):
    """
    Paso 5: Obtener la probabilidad P(var=valor | padres) desde la CPT.
    """
    tabla = cpts[var]

    # Filtrar las filas que coincidan con los valores de los padres y de la variable
    condicion = (tabla[var] == valor)
    for p in padres:
        condicion &= (tabla[p] == evidencia[p])

    prob = tabla.loc[condicion, "P"].values
    if len(prob) == 0:
        raise ValueError(f"No se encontró probabilidad para {var}={valor} con padres {padres}")
    return float(prob[0])

def normalizar(distribucion):
    """
    Paso 6: Normaliza las probabilidades para que sumen 1.
    """
    total = sum(distribucion.values())
    return {k: v / total for k, v in distribucion.items()}

def registrar_traza(mensaje, archivo="traza_inferencia.txt"):
    """
    Guarda un mensaje en un archivo de texto y lo muestra en consola.
    """
    print(mensaje)
    with open(archivo, "a", encoding="utf-8") as f:
        f.write(mensaje + "\n")

# === Cargar las tablas de probabilidad condicional (CPTs) ===
ruta_cpts = r"C:\Users\Catalina\OneDrive\Escritorio\proyecto_3_IA\Probabilidades"
grafo = cargar_grafo(r"C:\Users\Catalina\OneDrive\Escritorio\proyecto_3_IA\Dependencias_Estudiantes.csv")
cpts = cargar_cpts(ruta_cpts)

# === Generar los dominios automáticamente ===
dominios = generar_dominios(cpts)

print("\nDominios generados automáticamente:")
for var, valores in dominios.items():
    print(f"{var}: {valores}")

# === Definir la consulta y la evidencia ===
variable_objetivo, evidencia = definir_consulta_y_evidencia()
resultado = inferencia_por_enumeracion(variable_objetivo, evidencia, grafo, cpts, dominios)

print("\n=== Resultado de inferencia ===")
for val, prob in resultado.items():
    print(f"P({variable_objetivo}={val} | evidencia) = {prob:.4f}")