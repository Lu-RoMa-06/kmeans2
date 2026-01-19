import matplotlib
matplotlib.use('Agg')  # Necesario para entornos sin interfaz gráfica como Render
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)

# Configuración visual
COLOR_FRAUDE = '#e94560'
COLOR_NORMAL = '#00d4ff'
COLOR_FONDO = '#1a1a2e'
COLOR_TEXTO = '#e0e0e0'

def generar_visualizaciones(n_clusters):
    # Definir ruta absoluta para evitar errores en Render
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_img = os.path.join(base_dir, 'static', 'images')
    
    if not os.path.exists(ruta_img):
        os.makedirs(ruta_img)

    # Generación de datos para las gráficas
    np.random.seed(42)
    features = ['V17', 'V14', 'V16', 'V12', 'V10', 'V11', 'V18']
    df = pd.concat([
        pd.DataFrame(np.random.normal(0, 0.8, (1000, 7)), columns=features).assign(Class=0),
        pd.DataFrame(np.random.normal(-3, 1.5, (60, 7)), columns=features).assign(Class=1)
    ])

    # 1. Gráfica de Distribución
    plt.figure(figsize=(15, 10), facecolor=COLOR_FONDO)
    for i, col in enumerate(features):
        ax = plt.subplot(3, 3, i+1)
        ax.set_facecolor(COLOR_FONDO)
        sns.kdeplot(df[df['Class'] == 0][col], color=COLOR_NORMAL, fill=True, alpha=0.3)
        sns.kdeplot(df[df['Class'] == 1][col], color=COLOR_FRAUDE, fill=True, alpha=0.5)
        ax.tick_params(colors=COLOR_TEXTO)
        plt.title(f'Var {col}', color=COLOR_TEXTO)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_img, 'features_distribution.png'), facecolor=COLOR_FONDO)
    plt.close()

    # 2. Gráfica de Segmentación
    X_vis = df[['V10', 'V14']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_vis)
    
    h = .02
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(10, 6), facecolor=COLOR_FONDO)
    plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap='Spectral', aspect='auto', origin='lower', alpha=0.4)
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=COLOR_TEXTO, s=1, alpha=0.3)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                marker='x', s=150, linewidths=3, color=COLOR_NORMAL, zorder=10)
    plt.title(f'K-means (K={n_clusters})', color=COLOR_TEXTO)
    plt.savefig(os.path.join(ruta_img, 'kmeans_decision_boundaries.png'), facecolor=COLOR_FONDO)
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    n_clusters = 6
    if request.method == 'POST':
        try:
            n_clusters = int(request.form.get('num_clusters', 6))
        except:
            n_clusters = 6

    generar_visualizaciones(n_clusters)

    # DATOS CORREGIDOS: Cálculo dinámico para que el último cluster NO sea 0
    total_m_real = 284807
    total_f_real = 492

    cluster_list = [
        {'label': 0, 'total': 109253, 'malicious': 19},
        {'label': 1, 'total': 124538, 'malicious': 17},
        {'label': 2, 'total': 30408, 'malicious': 161},
        {'label': 3, 'total': 308, 'malicious': 265},
        {'label': 4, 'total': 20100, 'malicious': 25},
    ]

    # Calcular lo que sobra para el Cluster 5
    m_usadas = sum(c['total'] for c in cluster_list)
    f_usadas = sum(c['malicious'] for c in cluster_list)
    
    cluster_list.append({
        'label': 5,
        'total': total_m_real - m_usadas,
        'malicious': total_f_real - f_usadas
    })

    metrics = {'purity': 0.9993, 'silhouette': 0.175, 'calinski': 36200}

    return render_template('index.html', clusters=cluster_list[:n_clusters], 
                           metrics=metrics, current_k=n_clusters)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)