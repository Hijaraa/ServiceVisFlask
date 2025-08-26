from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)  # Pour permettre les requêtes cross-origin
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Créer le dossier uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Variable globale pour stocker le DataFrame
df_clients = None
ALLOWED_EXTENSIONS = {'csv'}

# FONCTIONS UTILITAIRES POUR CORRIGER L'ERREUR JSON
def convert_numpy_types(obj):
    """Convertit récursivement les types NumPy/Pandas en types Python natifs"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # Pour les scalaires NumPy
        return obj.item()
    else:
        return obj

def safe_jsonify(data):
    """Version sécurisée de jsonify qui gère les types NumPy"""
    cleaned_data = convert_numpy_types(data)
    return jsonify(cleaned_data)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_csv_data():
    """Charge le fichier CSV dans un DataFrame global"""
    global df_clients
    try:
        # Charger le fichier avec différents encodings possibles
        encodings = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df_clients = pd.read_csv('uploads/clients_data.csv', encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df_clients is None:
            raise Exception("Impossible de décoder le fichier CSV")
        
        # Nettoyer les données
        df_clients['phoneNumber'] = pd.to_numeric(df_clients['phoneNumber'], errors='coerce')
        
        # Supprimer les lignes complètement vides
        df_clients = df_clients.dropna(how='all')
        
        print(f"Données chargées: {len(df_clients)} clients")
        print(f"Colonnes détectées: {list(df_clients.columns)}")
        return True
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return False

@app.route('/', methods=['GET'])
def home():
    return safe_jsonify({
        "message": "Service Flask - Charts B2C Clients",
        "version": "1.0",
        "total_clients": len(df_clients) if df_clients is not None else 0,
        "endpoints": {
            "upload": "POST /upload - Uploader un fichier CSV",
            "pie_charts": {
                "gender_pie": "GET /api/charts/gender-pie",
                "cities_pie": "GET /api/charts/cities-pie?limit=10",
                "regions_pie": "GET /api/charts/regions-pie?limit=8",
                "relationship_pie": "GET /api/charts/relationship-pie",
                "workplace_pie": "GET /api/charts/workplace-pie?limit=10"
            },
            "bar_charts": {
                "top_cities_bar": "GET /api/charts/top-cities-bar?limit=15",
                "workplace_bar": "GET /api/charts/workplace-bar?limit=12",
                "countries_bar": "GET /api/charts/countries-bar?limit=10"
            },
            "advanced_charts": {
                "migration_analysis": "GET /api/charts/migration-analysis",
                "employment_stats": "GET /api/charts/employment-stats",
                "demographic_radar": "GET /api/charts/demographic-radar"
            },
            "dashboard": "GET /api/charts/dashboard-summary"
        }
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload et traite un fichier CSV"""
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'clients_data.csv')
        file.save(filepath)
        
        if load_csv_data():
            return safe_jsonify({
                'success': True,
                'message': f'Fichier uploadé et traité avec succès',
                'total_clients': len(df_clients)
            })
        else:
            return jsonify({'error': 'Erreur lors du traitement du fichier'}), 500
    
    return jsonify({'error': 'Format de fichier non supporté'}), 400

# ==================== PIE CHARTS B2C ====================

@app.route('/api/charts/gender-pie', methods=['GET'])
def gender_pie():
    """Données pour pie chart de la distribution par genre"""
    if df_clients is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    # Calculer les valeurs connues et inconnues
    gender_counts = df_clients['gender'].value_counts()
    
    # Compter les valeurs manquantes/vides
    unknown_count = df_clients['gender'].isna().sum() + (df_clients['gender'] == '').sum()
    
    total_clients = len(df_clients)
    
    # Préparer les données avec gestion des valeurs inconnues
    labels = []
    values = []
    colors = []
    
    # Mapping des couleurs par genre
    color_mapping = {
        'Female': "#FF4CB7",    # Rose pour femme
        'female': "#FF4CB7",    # Rose pour femme (minuscule)
        'F': '#FF4CB7',         # Rose pour femme (abréviation)
        'Male': "#0730e3",      # Rouge pour homme
        'male': '#0730e3',      # Rouge pour homme (minuscule) 
        'M': '#0730e3',         # Rouge pour homme (abréviation)
        'Other': "#4b5156",     # Gris pour autre
        'other': '#6c757d'      # Gris pour autre (minuscule)
    }
    
    # Ajouter les genres connus
    for gender, count in gender_counts.items():
        if pd.notna(gender) and gender != '':  # Exclure les valeurs vides
            labels.append(str(gender))
            values.append(int(count))
            # Utiliser la couleur mappée ou gris par défaut
            colors.append(color_mapping.get(gender, '#6c757d'))
    
    # Ajouter les valeurs inconnues si elles existent
    if unknown_count > 0:
        labels.append('Inconnu')
        values.append(int(unknown_count))
        colors.append('#868e96')  # Gris clair pour inconnu
    
    # Calculer les pourcentages
    percentages = [round((v/total_clients)*100, 1) for v in values]
    
    result = {
        'chart_type': 'pie',
        'title': 'Distribution par Genre',
        'data': {
            'labels': labels,
            'values': values,
            'total': total_clients
        },
        'percentages': percentages,
        'colors': colors,
        'statistics': {
            'known_values': total_clients - unknown_count,
            'unknown_values': unknown_count,
            'data_quality': round(((total_clients - unknown_count) / total_clients) * 100, 1)
        }
    }
    
    return safe_jsonify(result)

@app.route('/api/charts/cities-pie', methods=['GET'])
def cities_pie():
    """Données pour pie chart des villes"""
    if df_clients is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    limit = request.args.get('limit', 10, type=int)
    
    cities_count = df_clients['currentCity'].value_counts().head(limit)
    total_clients = len(df_clients)
    top_cities_total = int(cities_count.sum())
    others_count = total_clients - top_cities_total if limit < len(df_clients['currentCity'].value_counts()) else 0
    
    labels = cities_count.index.tolist()
    values = cities_count.values.tolist()
    
    if others_count > 0:
        labels.append('Autres villes')
        values.append(others_count)
    
    result = {
        'chart_type': 'pie',
        'title': f'Distribution par Ville (Top {limit})',
        'data': {
            'labels': labels,
            'values': values,
            'total': total_clients
        },
        'percentages': [round((v/total_clients)*100, 1) for v in values]
    }
    
    return safe_jsonify(result)

@app.route('/api/charts/regions-pie', methods=['GET'])
def regions_pie():
    """Données pour pie chart des régions"""
    if df_clients is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    limit = request.args.get('limit', 8, type=int)
    
    regions_count = df_clients['currentRegion'].value_counts().head(limit)
    total_clients = len(df_clients)
    
    result = {
        'chart_type': 'pie',
        'title': f'Distribution par Région (Top {limit})',
        'data': {
            'labels': regions_count.index.tolist(),
            'values': regions_count.values.tolist(),
            'total': total_clients
        },
        'percentages': [round((v/total_clients)*100, 1) for v in regions_count.values]
    }
    
    return safe_jsonify(result)

@app.route('/api/charts/relationship-pie', methods=['GET'])
def relationship_pie():
    """Données pour pie chart des statuts relationnels"""
    if df_clients is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    relationship_counts = df_clients['relationshipStatus'].value_counts()
    total_with_status = int(relationship_counts.sum())
    clients_without_status = len(df_clients) - total_with_status
    
    labels = relationship_counts.index.tolist()
    values = relationship_counts.values.tolist()
    
    if clients_without_status > 0:
        labels.append('Non spécifié')
        values.append(clients_without_status)
    
    total_clients = len(df_clients)
    
    result = {
        'chart_type': 'pie',
        'title': 'Statuts Relationnels',
        'data': {
            'labels': labels,
            'values': values,
            'total': total_clients
        },
        'percentages': [round((v/total_clients)*100, 1) for v in values],
        'colors': ['#28a745', '#dc3545', '#ffc107', '#6c757d']  # Vert, Rouge, Jaune, Gris
    }
    
    return safe_jsonify(result)

@app.route('/api/charts/workplace-pie', methods=['GET'])
def workplace_pie():
    """Données pour pie chart des lieux de travail"""
    if df_clients is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    limit = request.args.get('limit', 10, type=int)
    
    workplace_counts = df_clients['workplace'].value_counts().head(limit)
    total_clients = len(df_clients)
    clients_with_workplace = int(workplace_counts.sum())
    clients_without_workplace = total_clients - clients_with_workplace
    
    labels = workplace_counts.index.tolist()
    values = workplace_counts.values.tolist()
    
    if clients_without_workplace > 0:
        labels.append('Sans emploi spécifié')
        values.append(clients_without_workplace)
    
    result = {
        'chart_type': 'pie',
        'title': f'Lieux de Travail (Top {limit})',
        'data': {
            'labels': labels,
            'values': values,
            'total': total_clients
        },
        'percentages': [round((v/total_clients)*100, 1) for v in values]
    }
    
    return safe_jsonify(result)

# ==================== BAR CHARTS B2C ====================
# 🏙️ BAR CHART VERTICAL - TOP VILLES ACTUELLES
@app.route('/api/charts/top-cities-bar', methods=['GET'])
def top_cities_bar():
    """Données pour bar chart vertical des top villes actuelles"""
    if df_clients is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
   
    limit = request.args.get('limit', 15, type=int)
   
    # Utiliser current_city (nom normalisé) ou currentCity selon le format
    city_column = 'current_city' if 'current_city' in df_clients.columns else 'currentCity'
    
    if city_column not in df_clients.columns:
        return jsonify({'error': f'Colonne {city_column} non trouvée'}), 400
    
    # Filtrer les valeurs non nulles et non vides
    cities_data = df_clients[df_clients[city_column].notna() & (df_clients[city_column] != '')]
    cities_count = cities_data[city_column].value_counts().head(limit)
   
    result = {
        'chart_type': 'vertical_bar',  # 🆕 Type vertical
        'title': f'Top {limit} Villes Actuelles par Nombre de Clients',
        'data': {
            'labels': cities_count.index.tolist(),
            'values': cities_count.values.tolist()
        },
        'axis_labels': {
            'x': 'Nombre de Clients',  # 🔄 Inversé pour vertical
            'y': 'Villes'              # 🔄 Inversé pour vertical
        },
        'orientation': 'vertical',     # 🆕 Indication d'orientation
        'total_cities': len(cities_count),
        'total_clients_with_city': len(cities_data)
    }
   
    return safe_jsonify(result)

@app.route('/api/charts/workplace-bar', methods=['GET'])
def workplace_bar():
    """Données pour bar chart des lieux de travail"""
    if df_clients is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    limit = request.args.get('limit', 12, type=int)
    
    workplace_counts = df_clients['workplace'].value_counts().head(limit)
    
    result = {
        'chart_type': 'bar',
        'title': f'Top {limit} Lieux de Travail',
        'data': {
            'labels': workplace_counts.index.tolist(),
            'values': workplace_counts.values.tolist()
        },
        'axis_labels': {
            'x': 'Lieux de Travail',
            'y': 'Nombre de Clients'
        }
    }
    
    return safe_jsonify(result)

@app.route('/api/charts/countries-bar', methods=['GET'])
def countries_bar():
    """Données pour bar chart des pays"""
    if df_clients is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    limit = request.args.get('limit', 10, type=int)
    
    countries_count = df_clients['currentCountry'].value_counts().head(limit)
    
    result = {
        'chart_type': 'bar',
        'title': f'Top {limit} Pays',
        'data': {
            'labels': countries_count.index.tolist(),
            'values': countries_count.values.tolist()
        },
        'axis_labels': {
            'x': 'Pays',
            'y': 'Nombre de Clients'
        }
    }
    
    return safe_jsonify(result)

# ==================== ADVANCED CHARTS B2C ====================

@app.route('/api/charts/migration-analysis', methods=['GET'])
def migration_analysis():
    """Analyse de migration - clients qui ont déménagé"""
    if df_clients is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    # Clients avec ville d'origine
    clients_with_hometown = df_clients[df_clients['hometownCity'].notna()]
    
    # Clients ayant migré
    if len(clients_with_hometown) > 0:
        migrants = clients_with_hometown[
            clients_with_hometown['currentCity'] != clients_with_hometown['hometownCity']
        ]
        migration_rate = len(migrants) / len(clients_with_hometown) * 100
    else:
        migrants = pd.DataFrame()
        migration_rate = 0
    
    # Analyse des pays
    clients_with_home_country = df_clients[df_clients['hometownCountry'].notna()]
    if len(clients_with_home_country) > 0:
        international_migrants = clients_with_home_country[
            clients_with_home_country['currentCountry'] != clients_with_home_country['hometownCountry']
        ]
        international_migration_rate = len(international_migrants) / len(clients_with_home_country) * 100
    else:
        international_migrants = pd.DataFrame()
        international_migration_rate = 0
    
    result = {
        'chart_type': 'bar',
        'title': 'Analyse de Migration',
        'data': {
            'labels': ['Restés dans la même ville', 'Ont déménagé (ville)', 'Restés dans le même pays', 'Ont émigré (pays)'],
            'values': [
                len(clients_with_hometown) - len(migrants),
                len(migrants),
                len(clients_with_home_country) - len(international_migrants),
                len(international_migrants)
            ]
        },
        'migration_stats': {
            'total_clients': len(df_clients),
            'clients_with_hometown_data': len(clients_with_hometown),
            'internal_migration_rate': round(migration_rate, 1),
            'international_migration_rate': round(international_migration_rate, 1)
        }
    }
    
    return safe_jsonify(result)

@app.route('/api/charts/employment-stats', methods=['GET'])
def employment_stats():
    """Statistiques d'emploi détaillées"""
    if df_clients is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    total_clients = len(df_clients)
    clients_with_workplace = df_clients['workplace'].notna().sum()
    employment_rate = (clients_with_workplace / total_clients) * 100
    
    # Analyse des secteurs
    employment_sectors = {
        'Santé': df_clients['workplace'].str.contains('hospital|hopital|medical|dentaire|health', case=False, na=False).sum(),
        'Mode/Retail': df_clients['workplace'].str.contains('h&m|gucci|fashion|retail|store', case=False, na=False).sum(),
        'Auto-entrepreneurs': df_clients['workplace'].str.contains('self-employed|freelance|independent', case=False, na=False).sum(),
        'Corporate': df_clients['workplace'].str.contains('corp|company|enterprise|inc', case=False, na=False).sum(),
        'Tech': df_clients['workplace'].str.contains('tech|software|digital|IT|computer', case=False, na=False).sum()
    }
    
    result = {
        'chart_type': 'bar',
        'title': 'Statistiques d\'Emploi par Secteur',
        'data': {
            'labels': list(employment_sectors.keys()),
            'values': list(employment_sectors.values())
        },
        'employment_overview': {
            'total_clients': total_clients,
            'employed': int(clients_with_workplace),
            'unemployed_or_unspecified': total_clients - int(clients_with_workplace),
            'employment_rate': round(employment_rate, 1)
        }
    }
    
    return safe_jsonify(result)

@app.route('/api/charts/demographic-radar', methods=['GET'])
def demographic_radar():
    """Radar chart des caractéristiques démographiques par genre"""
    if df_clients is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    # Analyser par genre
    gender_stats = {}
    genders = df_clients['gender'].dropna().unique()
    
    for gender in genders:
        gender_data = df_clients[df_clients['gender'] == gender]
        
        # Calculer les métriques (normalisées sur 100)
        employment_rate = (gender_data['workplace'].notna().sum() / len(gender_data)) * 100
        relationship_rate = (gender_data['relationshipStatus'].notna().sum() / len(gender_data)) * 100
        migration_rate = 0
        if gender_data['hometownCity'].notna().sum() > 0:
            migrants = gender_data[
                (gender_data['hometownCity'].notna()) & 
                (gender_data['currentCity'] != gender_data['hometownCity'])
            ]
            migration_rate = (len(migrants) / gender_data['hometownCity'].notna().sum()) * 100
        
        diversity_rate = (gender_data['currentCity'].nunique() / len(gender_data)) * 100
        
        gender_stats[gender] = {
            'employment': round(employment_rate, 1),
            'relationship': round(relationship_rate, 1),
            'migration': round(migration_rate, 1),
            'geographic_diversity': round(min(diversity_rate, 100), 1)  # Cap à 100%
        }
    
    result = {
        'chart_type': 'radar',
        'title': 'Profil Démographique par Genre',
        'data': {
            'categories': list(genders),
            'metrics': ['Taux d\'emploi', 'Info relationnelle', 'Mobilité', 'Diversité géographique'],
            'values': [
                [gender_stats[gender]['employment'], gender_stats[gender]['relationship'], 
                 gender_stats[gender]['migration'], gender_stats[gender]['geographic_diversity']]
                for gender in genders
            ]
        }
    }
    
    return safe_jsonify(result)

# ==================== DASHBOARD SUMMARY B2C ====================

@app.route('/api/charts/dashboard-summary', methods=['GET'])
def dashboard_summary():
    """Données résumées pour dashboard B2C"""
    if df_clients is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    # KPIs principaux
    total_clients = len(df_clients)
    gender_distribution = df_clients['gender'].value_counts().to_dict()
    
    # Calculer les métriques
    employment_rate = (df_clients['workplace'].notna().sum() / total_clients) * 100
    
    # Clients avec informations de migration
    clients_with_hometown = df_clients['hometownCity'].notna().sum()
    migration_rate = 0
    if clients_with_hometown > 0:
        migrants = df_clients[
            (df_clients['hometownCity'].notna()) & 
            (df_clients['currentCity'] != df_clients['hometownCity'])
        ]
        migration_rate = (len(migrants) / clients_with_hometown) * 100
    
    # Top données pour les graphiques
    top_cities = {str(k): int(v) for k, v in df_clients['currentCity'].value_counts().head(5).to_dict().items()}
    top_workplaces = {str(k): int(v) for k, v in df_clients['workplace'].value_counts().head(5).to_dict().items()}
    relationship_distribution = {str(k): int(v) for k, v in df_clients['relationshipStatus'].value_counts().to_dict().items()}
    
    result = {
        'kpis': {
            'total_clients': int(total_clients),
            'employment_rate': round(float(employment_rate), 1),
            'migration_rate': round(float(migration_rate), 1),
            'unique_cities': int(df_clients['currentCity'].nunique()),
            'unique_workplaces': int(df_clients['workplace'].nunique()),
            'unique_countries': int(df_clients['currentCountry'].nunique())
        },
        'distributions': {
            'gender': {str(k): int(v) for k, v in gender_distribution.items()},
            'relationship': relationship_distribution,
            'cities': top_cities,
            'workplaces': top_workplaces
        },
        'data_quality': {
            'clients_with_workplace': int(df_clients['workplace'].notna().sum()),
            'clients_with_relationship_status': int(df_clients['relationshipStatus'].notna().sum()),
            'clients_with_hometown': int(df_clients['hometownCity'].notna().sum()),
            'clients_with_phone': int(df_clients['phoneNumber'].notna().sum())
        }
    }
    
    return safe_jsonify(result)

if __name__ == '__main__':
    # Essayer de charger les données au démarrage si le fichier existe
    if os.path.exists('uploads/clients_data.csv'):
        load_csv_data()
    
    app.run(debug=True, host='0.0.0.0', port=5002)  # Port 5001 pour B2C