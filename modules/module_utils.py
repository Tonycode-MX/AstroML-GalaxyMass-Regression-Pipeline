import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import os
from typing import Optional

def save_dataframe(df, data_folder, filename="dataframe.csv"):
    """
    Saves the DataFrame to a CSV file at the specified path.
    Args:
        df (pd.DataFrame): DataFrame to save.
        data_folder (str): Path to the folder where the file will be saved.
        filename (str): Name of the output file (default is 'dataframe.csv').
    Returns:
        None
    """
    save_path = os.path.join(data_folder, filename)
    df.to_csv(save_path, index=False)
    print(f"\nDataFrame saved to {save_path}")

def save_list_to_file(list_to_save, folder_path, filename="list.pkl"):
    """
    Saves a list of items to a pickle file, one item per line.
    Args:
        list_to_save (list): List of items to save.
        folder_path (str): Path to the folder where the file will be saved.
        filename (str): Name of the output file (default is 'list.pkl').
    Returns:
        None
    """
    os.makedirs(folder_path, exist_ok=True)
    full_path = os.path.join(folder_path, filename)
    with open(full_path, 'wb') as file:
        pickle.dump(list_to_save, file)

    print(f"\nList saved as '{filename}' successfully.")


def load_list_from_file(folder_path, filename="list.pkl"):
    """
    Loads a list from a pickle file at a specified path and returns it.
    
    Args:
        folder_path (str): The directory where the file is located.
        filename (str): The name of the pickle file.
        
    Returns:
        list: The list loaded from the file, or None if the file is not found.
    """
    # build full file path
    full_path = os.path.join(folder_path, filename)
    
    # load list from pickle file
    try:
        with open(full_path, 'rb') as file:
            loaded_list = pickle.load(file)
        
        print(f"\nList loaded from '{full_path}' successfully.")
        return loaded_list
    
    except FileNotFoundError:
        print(f"\nError: The file '{full_path}' was not found.")
        return None

def reorder_and_rename_target(df: pd.DataFrame, old_col_name="target", new_col_name="target") -> pd.DataFrame:
    """
    Moves the target column to the last position and renames it to 'target'.

    Args:
        df (pd.DataFrame): The original DataFrame containing the target column.

    Returns:
        pd.DataFrame: The modified DataFrame with the 'target' column at the end.
    """
    df = df.rename(columns={old_col_name: new_col_name})
    cols = df.columns.tolist()
    cols.remove(new_col_name)
    cols.append(new_col_name)
    df = df[cols]

    return df

def plot_feature_vs_target_scatter(
    df: pd.DataFrame, 
    target_col: str = 'target', 
    save_dir: Optional[str] = None, 
    filename: Optional[str] = None,
    sample_size: Optional[int] = 10000,
    point_alpha: float = 0.2
):
    """
    Genera diagramas de dispersión optimizados para menor complejidad y peso en PDF.

    Implementa: 
    1. Submuestreo (si sample_size es menor que la longitud del DF) para reducir la cantidad de puntos a dibujar.
    2. Rasterización (`rasterized=True`) de los puntos para forzar su representación como un mapa de bits
       en lugar de objetos vectoriales individuales dentro del PDF, aligerando el archivo.
    3. Transparencia (`point_alpha`) para mejorar la visualización de la densidad.
    
    Args:
        df (pd.DataFrame): DataFrame de Pandas que contiene las características y el objetivo.
        target_col (str): Nombre de la columna de la variable objetivo. Por defecto es 'target'.
        save_dir (str, optional): Directorio donde guardar el archivo.
        filename (str, optional): Nombre del archivo a guardar.
        sample_size (int, optional): Máximo número de puntos a muestrear y plotear. Por defecto es 100000.
        point_alpha (float): Transparencia de los puntos de dispersión (0.0 a 1.0). Por defecto es 0.2.
    """
    
    # --- 0. Manejo de Submuestreo ---
    df_plot = df
    if sample_size is not None and len(df) > sample_size:
        # Usamos .sample() para submuestrear de forma aleatoria y representativa
        df_plot = df.sample(n=sample_size, random_state=42)
        print(f"Advertencia: El DataFrame se submuestrea a {sample_size} puntos para aligerar el gráfico.")
        
    # --- 1. Preparación de datos y diseño de la cuadrícula ---
    feature_cols = df_plot.columns.drop(target_col, errors='ignore')
    if target_col not in df_plot.columns:
        print(f"Error: La columna objetivo '{target_col}' no se encuentra en el DataFrame.")
        return
        
    n_features = len(feature_cols)
    if n_features == 0:
        print("Advertencia: No hay columnas de características para graficar.")
        return

    n_rows = (n_features + 2) // 3
    n_cols = 3 if n_features >= 3 else n_features or 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    # --- 2. Generación de los gráficos de dispersión (OPTIMIZACIÓN CLAVE) ---
    for i, col in enumerate(feature_cols):
        try:
            # 🔑 OPTIMIZACIÓN 1: rasterized=True
            # Esto convierte los puntos en un mapa de bits de alta resolución dentro del PDF,
            # lo que reduce la complejidad vectorial del archivo final.
            sns.scatterplot(
                x=df_plot[col], 
                y=df_plot[target_col], 
                ax=axes[i], 
                alpha=point_alpha,
                rasterized=True 
            )
            axes[i].set_title(f'{col} vs. {target_col}', fontsize=10)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel(target_col)
        except Exception as e:
            print(f"Error al graficar la columna '{col}': {e}")

    # --- 3. Limpieza de ejes vacíos ---
    for j in range(n_features, len(axes)):
        if hasattr(fig, 'delaxes'):
            fig.delaxes(axes[j])
            
    plt.tight_layout()
    
    # --- 4. Guardar el gráfico ---
    if save_dir and filename:
        full_path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True) 
        
        try:
            # 🔑 OPTIMIZACIÓN 2: Usar dpi alto (ej. 300) en savefig
            # Asegura que la imagen rasterizada tenga buena calidad.
            plt.savefig(full_path, format='pdf', bbox_inches='tight', dpi=300) 
            print(f"Gráfico de dispersión optimizado guardado en: {full_path}")
        except Exception as e:
            print(f"Error al guardar el archivo PDF: {e}")
            
    # --- 5. Cierre de la figura ---
    plt.close(fig)

def plot_boxplots(
    df: pd.DataFrame, 
    target_col: str = 'target', 
    save_dir: Optional[str] = None, 
    filename: Optional[str] = None
):
    """
    Genera diagramas de caja optimizados (aligerados) para visualizar outliers.
    
    Se elimina el argumento 'rasterized' de sns.boxplot y se aplica a los elementos
    'fliers' (outliers) después de la creación del gráfico para evitar el TypeError.
    """
    # Se grafican todas las columnas, incluyendo el target
    cols = df.columns.tolist() 
    n_cols = len(cols)
    
    # Ajuste dinámico de la cuadrícula
    n_rows = (n_cols + 2) // 3
    n_cols_plot = 3
    
    fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        
        # 1. Generar el Box Plot (SIN rasterized)
        sns.boxplot(
            y=df[col], 
            ax=axes[i],
            fliersize=3       # Controlar el tamaño de los outliers
            # Se omite 'rasterized=True' para evitar el TypeError
        )
        axes[i].set_title(f'Box Plot de {col}')
        axes[i].set_ylabel(col)
        
        # 2. 🔑 SOLUCIÓN: Aplicar rasterización SOLO a los outliers (fliers)
        # Los outliers son los elementos que causan la pesadez vectorial.
        try:
            # Obtener los elementos de la gráfica (líneas, cajas, fliers)
            box_artists = axes[i].get_children()
            # Los 'fliers' (outliers) son generalmente los últimos elementos dibujados (plt.Line2D)
            # Buscamos los PathCollection o Line2D que representan los fliers
            for item in box_artists:
                if isinstance(item, plt.Line2D) or item.__class__.__name__ == 'PathCollection':
                    # Aplicar rasterización a los fliers
                    item.set_rasterized(True)
        except Exception:
            # Continuar si no se pueden obtener o rasterizar los elementos
            pass 

    # Limpieza de ejes vacíos
    for j in range(n_cols, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()

    # Guardar el gráfico
    if save_dir and filename:
        full_path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Usar dpi alto para la calidad del mapa de bits
            plt.savefig(full_path, format='pdf', bbox_inches='tight', dpi=300)
            print(f"Gráfico de caja optimizado guardado en: {full_path}")
        except Exception as e:
            print(f"Error al guardar el archivo PDF: {e}")
            
    # Liberación de memoria
    plt.close(fig)

def plot_correlation_heatmap(
    df: pd.DataFrame, 
    save_dir: Optional[str] = None, 
    filename: Optional[str] = None
):
    """
    Genera un mapa de calor para visualizar la matriz de correlación entre todas las variables.
    Optimizado para un PDF más ligero mediante rasterización.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        save_dir (str, optional): Directorio donde guardar el archivo.
        filename (str, optional): Nombre del archivo a guardar.
    """
    
    # 1. Calcular la matriz de correlación
    corr_matrix = df.corr()
    
    # Máscara para ocultar la mitad superior (es redundante)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(12, 10))
    
    # 2. Generar el Heatmap
    heatmap = sns.heatmap(
        corr_matrix, 
        mask=mask, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        linewidths=.5, 
        cbar_kws={"shrink": .8}
    )
    plt.title('Mapa de Calor de la Matriz de Correlación', fontsize=16)

    # 🔑 OPTIMIZACIÓN CLAVE: Rasterizar el lienzo del heatmap
    # Esto asegura que el área de color (la matriz) se guarde como un mapa de bits.
    heatmap.get_figure().patch.set_rasterized(True)

    # El texto de las anotaciones se mantiene vectorial para que sea nítido.
    
    # 3. Guardar el gráfico
    if save_dir and filename:
        full_path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Usar dpi alto para la calidad del mapa de bits y la opción PDF
            plt.savefig(full_path, format='pdf', bbox_inches='tight', dpi=300)
            print(f"Mapa de calor optimizado guardado en: {full_path}")
        except Exception as e:
            print(f"Error al guardar el archivo PDF: {e}")
            
    # 4. Liberación de memoria
    plt.close() 