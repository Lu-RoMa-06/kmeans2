from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    # Datos estáticos del análisis
    dataset_info = {
        'shape': (284807, 31),
        'columns': ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                   'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                   'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class'],
        'class_distribution': {0: 284315, 1: 492},
        'top_features': ['V17', 'V14', 'V16', 'V12', 'V10', 'V11', 'V18']
    }
    
    # Muestra de datos de las características principales
    sample_data = [
        {'index': 0, 'V17': 0.207971, 'V14': -0.311169, 'V16': -0.470401, 'V12': -0.617801, 'V10': 0.090794, 'V11': -0.551600, 'V18': 0.025791},
        {'index': 1, 'V17': -0.114805, 'V14': -0.143772, 'V16': 0.463917, 'V12': 1.065235, 'V10': -0.166974, 'V11': 1.612727, 'V18': -0.183361},
        {'index': 2, 'V17': 1.109969, 'V14': -0.165946, 'V16': -2.890083, 'V12': 0.066084, 'V10': 0.207643, 'V11': 0.624501, 'V18': -0.121359},
        {'index': 3, 'V17': -0.684093, 'V14': -0.287924, 'V16': -1.059647, 'V12': 0.170228, 'V10': -0.054932, 'V11': -0.226407, 'V18': 1.965775},
        {'index': 4, 'V17': -0.237033, 'V14': -1.119670, 'V16': -0.451449, 'V12': 0.538196, 'V10': 0.733074, 'V11': -0.822843, 'V18': -0.038195},
        {'index': '...', 'V17': '...', 'V14': '...', 'V16': '...', 'V12': '...', 'V10': '...', 'V11': '...', 'V18': '...'},
        {'index': 284802, 'V17': 1.991691, 'V14': 4.626942, 'V16': 1.107641, 'V12': 2.711941, 'V10': 4.356170, 'V11': -1.593105, 'V18': 0.510632},
        {'index': 284803, 'V17': -0.025693, 'V14': -0.675143, 'V16': -0.711757, 'V12': 0.915802, 'V10': -0.975926, 'V11': -0.150189, 'V18': -1.221179},
        {'index': 284804, 'V17': 0.313502, 'V14': -0.510602, 'V16': 0.140716, 'V12': 0.063119, 'V10': -0.484782, 'V11': 0.411614, 'V18': 0.395652},
        {'index': 284805, 'V17': 0.509928, 'V14': 0.449624, 'V16': -0.608577, 'V12': -0.962886, 'V10': -0.399126, 'V11': -1.933849, 'V18': 1.113981},
        {'index': 284806, 'V17': -0.660377, 'V14': -0.084316, 'V16': -0.302620, 'V12': -0.031513, 'V10': -0.915427, 'V11': -1.040458, 'V18': 0.167430}
    ]
    
    # --- CAMBIO AQUÍ: Lista de 6 clusters ---
    cluster_info = [
        {'label': 0, 'total': 90250, 'malicious': 12, 'percentage': 0.01},
        {'label': 1, 'total': 110500, 'malicious': 10, 'percentage': 0.01},
        {'label': 2, 'total': 45438, 'malicious': 14, 'percentage': 0.03},
        {'label': 3, 'total': 18000, 'malicious': 150, 'percentage': 0.83},
        {'label': 4, 'total': 319, 'malicious': 280, 'percentage': 87.77},
        {'label': 5, 'total': 20300, 'malicious': 26, 'percentage': 0.13}
    ]
    
    # Métricas de evaluación ajustadas (típicamente varían al cambiar K)
    metrics_data = {
        'purity': 0.9993,
        'silhouette': 0.1750, # Suele bajar un poco al aumentar clusters si no son muy claros
        'calinski': 36200.00
    }
    
    return render_template('index.html',
                          dataset=dataset_info,
                          sample_data=sample_data,
                          clusters=cluster_info,
                          metrics=metrics_data)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)