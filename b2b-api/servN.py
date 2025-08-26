from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
import json
from collections import Counter

app = Flask(__name__)
CORS(app)  # Pour permettre les requêtes cross-origin
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Créer le dossier uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Variable globale pour stocker le DataFrame
df_companies = None
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
    global df_companies
    try:
        # Charger le fichier avec différents encodings possibles
        encodings = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df_companies = pd.read_csv('uploads/companies_data.csv', encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df_companies is None:
            raise Exception("Impossible de décoder le fichier CSV")
        
        # Nettoyer les données
        df_companies['Rating'] = pd.to_numeric(df_companies['Rating'], errors='coerce')
        df_companies['Reviews'] = pd.to_numeric(df_companies['Reviews'], errors='coerce')
        df_companies['Score'] = pd.to_numeric(df_companies['Score'], errors='coerce')
        
        # Supprimer les lignes avec des valeurs manquantes critiques
        df_companies = df_companies.dropna(subset=['Name'])
        
        print(f"Données chargées: {len(df_companies)} entreprises")
        return True
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return False

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Service Flask - Statistiques pour Charts B2B",
        "version": "1.0",
        "total_companies": len(df_companies) if df_companies is not None else 0,
        "endpoints": {
            "upload": "POST /upload - Uploader un fichier CSV",
            "pie_charts": {
                "categories_pie": "GET /api/charts/categories-pie?limit=10",
                "cities_pie": "GET /api/charts/cities-pie?limit=15",
                "score_categories_pie": "GET /api/charts/score-categories-pie",
                "countries_pie": "GET /api/charts/countries-pie?limit=10"
            },
            "bar_charts": {
                "top_cities_bar": "GET /api/charts/top-cities-bar?limit=20",
                "categories_bar": "GET /api/charts/categories-bar?limit=15",
                "ratings_distribution_bar": "GET /api/charts/ratings-distribution-bar",
                "scores_distribution_bar": "GET /api/charts/scores-distribution-bar"
            },
            "line_charts": {
                "rating_trends": "GET /api/charts/rating-trends",
                "score_trends": "GET /api/charts/score-trends"
            },
            "advanced_charts": {
                "rating_vs_reviews_scatter": "GET /api/charts/rating-vs-reviews-scatter",
                "categories_performance_radar": "GET /api/charts/categories-performance-radar?limit=8",
                "geographic_heatmap": "GET /api/charts/geographic-heatmap"
            }
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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'companies_data.csv')
        file.save(filepath)
        
        if load_csv_data():
            return jsonify({
                'success': True,
                'message': f'Fichier uploadé et traité avec succès',
                'total_companies': len(df_companies)
            })
        else:
            return jsonify({'error': 'Erreur lors du traitement du fichier'}), 500
    
    return jsonify({'error': 'Format de fichier non supporté'}), 400

# ==================== PIE CHARTS ====================

@app.route('/api/charts/categories-pie', methods=['GET'])
def categories_pie():
    """Données pour pie chart des catégories d'entreprises"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    limit = request.args.get('limit', 10, type=int)
    
    # Compter les catégories
    categories_count = df_companies['Main Category'].value_counts().head(limit)
    
    # Calculer le reste si limité
    total_companies = len(df_companies)
    top_categories_total = int(categories_count.sum())
    others_count = total_companies - top_categories_total if limit < len(df_companies['Main Category'].value_counts()) else 0
    
    # Préparer les données pour le pie chart - CONVERSION EXPLICITE
    labels = [str(label) for label in categories_count.index.tolist()]
    values = [int(val) for val in categories_count.values.tolist()]
    
    if others_count > 0:
        labels.append('Autres')
        values.append(int(others_count))
    
    result = {
        'chart_type': 'pie',
        'title': f'Distribution des Entreprises par Catégorie (Top {limit})',
        'data': {
            'labels': labels,
            'values': values,
            'total': int(total_companies)
        },
        'percentages': [round((v/total_companies)*100, 1) for v in values]
    }
    
    return safe_jsonify(result)

@app.route('/api/charts/cities-pie', methods=['GET'])
def cities_pie():
    """Données pour pie chart des villes"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    limit = request.args.get('limit', 15, type=int)
    
    cities_count = df_companies['City'].value_counts().head(limit)
    total_companies = len(df_companies)
    top_cities_total = int(cities_count.sum())
    others_count = total_companies - top_cities_total if limit < len(df_companies['City'].value_counts()) else 0
    
    labels = [str(label) for label in cities_count.index.tolist()]
    values = [int(val) for val in cities_count.values.tolist()]
    
    if others_count > 0:
        labels.append('Autres villes')
        values.append(int(others_count))
    
    result = {
        'chart_type': 'pie',
        'title': f'Distribution des Entreprises par Ville (Top {limit})',
        'data': {
            'labels': labels,
            'values': values,
            'total': int(total_companies)
        },
        'percentages': [round((v/total_companies)*100, 1) for v in values]
    }
    
    return jsonify(result)

@app.route('/api/charts/score-categories-pie', methods=['GET'])
def score_categories_pie():
    """Données pour pie chart des catégories de score"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    score_categories = df_companies['Score Category'].value_counts()
    total_with_score = int(score_categories.sum())
    
    result = {
        'chart_type': 'pie',
        'title': 'Distribution par Catégorie de Score',
        'data': {
            'labels': [str(label) for label in score_categories.index.tolist()],
            'values': [int(val) for val in score_categories.values.tolist()],
            'total': total_with_score
        },
        'percentages': [round((int(v)/total_with_score)*100, 1) for v in score_categories.values],
        'colors': ['#28a745', '#ffc107', '#fd7e14', '#dc3545']  # Vert, Jaune, Orange, Rouge
    }
    
    return jsonify(result)

@app.route('/api/charts/countries-pie', methods=['GET'])
def countries_pie():
    """Données pour pie chart des pays"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    limit = request.args.get('limit', 10, type=int)
    
    countries_count = df_companies['Country Code'].value_counts().head(limit)
    total_companies = len(df_companies)
    
    result = {
        'chart_type': 'pie',
        'title': f'Distribution des Entreprises par Pays (Top {limit})',
        'data': {
            'labels': countries_count.index.tolist(),
            'values': countries_count.values.tolist(),
            'total': total_companies
        },
        'percentages': [round((v/total_companies)*100, 1) for v in countries_count.values]
    }
    
    return safe_jsonify(result)

# ==================== BAR CHARTS ====================

@app.route('/api/charts/top-cities-bar', methods=['GET'])
def top_cities_bar():
    """Données pour bar chart des top villes"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    limit = request.args.get('limit', 20, type=int)
    
    cities_count = df_companies['City'].value_counts().head(limit)
    
    result = {
        'chart_type': 'bar',
        'title': f'Top {limit} Villes par Nombre d\'Entreprises',
        'data': {
            'labels': cities_count.index.tolist(),
            'values': cities_count.values.tolist()
        },
        'axis_labels': {
            'x': 'Villes',
            'y': 'Nombre d\'Entreprises'
        }
    }
    
    return safe_jsonify(result)

@app.route('/api/charts/categories-bar', methods=['GET'])
def categories_bar():
    """Données pour bar chart des catégories"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    limit = request.args.get('limit', 15, type=int)
    
    categories_count = df_companies['Main Category'].value_counts().head(limit)
    
    result = {
        'chart_type': 'bar',
        'title': f'Top {limit} Catégories d\'Entreprises',
        'data': {
            'labels': categories_count.index.tolist(),
            'values': categories_count.values.tolist()
        },
        'axis_labels': {
            'x': 'Catégories',
            'y': 'Nombre d\'Entreprises'
        }
    }
    
    return jsonify(result)

@app.route('/api/charts/ratings-distribution-bar', methods=['GET'])
def ratings_distribution_bar():
    """Données pour bar chart de la distribution des ratings"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    # Créer des tranches de rating
    ratings_with_data = df_companies[df_companies['Rating'].notna()]
    
    # Créer des bins pour les ratings
    bins = [0, 1, 2, 3, 4, 5]
    labels = ['0-1', '1-2', '2-3', '3-4', '4-5']
    
    ratings_binned = pd.cut(ratings_with_data['Rating'], bins=bins, labels=labels, include_lowest=True)
    rating_counts = ratings_binned.value_counts().sort_index()
    
    result = {
        'chart_type': 'bar',
        'title': 'Distribution des Ratings',
        'data': {
            'labels': rating_counts.index.tolist(),
            'values': rating_counts.values.tolist()
        },
        'axis_labels': {
            'x': 'Plages de Rating',
            'y': 'Nombre d\'Entreprises'
        },
        'total_rated_companies': len(ratings_with_data)
    }
    
    return jsonify(result)

@app.route('/api/charts/scores-distribution-bar', methods=['GET'])
def scores_distribution_bar():
    """Données pour bar chart de la distribution des scores"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    scores_with_data = df_companies[df_companies['Score'].notna()]
    
    # Créer des bins pour les scores
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
    
    scores_binned = pd.cut(scores_with_data['Score'], bins=bins, labels=labels, include_lowest=True)
    score_counts = scores_binned.value_counts().sort_index()
    
    result = {
        'chart_type': 'bar',
        'title': 'Distribution des Scores',
        'data': {
            'labels': score_counts.index.tolist(),
            'values': score_counts.values.tolist()
        },
        'axis_labels': {
            'x': 'Plages de Score',
            'y': 'Nombre d\'Entreprises'
        },
        'total_scored_companies': len(scores_with_data)
    }
    
    return jsonify(result)

# ==================== LINE CHARTS ====================

@app.route('/api/charts/rating-trends', methods=['GET'])
def rating_trends():
    """Données pour line chart des tendances de rating par catégorie"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    # Calculer rating moyen par catégorie
    rating_by_category = df_companies.groupby('Main Category')['Rating'].mean().sort_values(ascending=False).head(10)
    
    result = {
        'chart_type': 'line',
        'title': 'Rating Moyen par Catégorie (Top 10)',
        'data': {
            'labels': rating_by_category.index.tolist(),
            'values': [round(val, 2) for val in rating_by_category.values.tolist()]
        },
        'axis_labels': {
            'x': 'Catégories',
            'y': 'Rating Moyen'
        }
    }
    
    return jsonify(result)

@app.route('/api/charts/score-trends', methods=['GET'])
def score_trends():
    """Données pour line chart des tendances de score par ville"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    # Calculer score moyen par ville (top 15)
    score_by_city = df_companies.groupby('City')['Score'].mean().sort_values(ascending=False).head(15)
    
    result = {
        'chart_type': 'line',
        'title': 'Score Moyen par Ville (Top 15)',
        'data': {
            'labels': score_by_city.index.tolist(),
            'values': [round(val, 2) for val in score_by_city.values.tolist()]
        },
        'axis_labels': {
            'x': 'Villes',
            'y': 'Score Moyen'
        }
    }
    
    return jsonify(result)

# ==================== ADVANCED CHARTS ====================

@app.route('/api/charts/rating-vs-reviews-scatter', methods=['GET'])
def rating_vs_reviews_scatter():
    """Données pour scatter plot Rating vs Reviews"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    # Filtrer les données avec rating et reviews
    scatter_data = df_companies[(df_companies['Rating'].notna()) & (df_companies['Reviews'].notna())]
    
    # Limiter pour éviter trop de points
    if len(scatter_data) > 1000:
        scatter_data = scatter_data.sample(1000)
    
    result = {
        'chart_type': 'scatter',
        'title': 'Relation Rating vs Nombre de Reviews',
        'data': {
            'points': [
                {
                    'x': int(row['Reviews']),
                    'y': float(row['Rating']),
                    'name': row['Name'][:30] + '...' if len(row['Name']) > 30 else row['Name']
                }
                for _, row in scatter_data.iterrows()
            ]
        },
        'axis_labels': {
            'x': 'Nombre de Reviews',
            'y': 'Rating'
        },
        'total_points': len(scatter_data)
    }
    
    return jsonify(result)

@app.route('/api/charts/categories-performance-radar', methods=['GET'])
def categories_performance_radar():
    """Données pour radar chart des performances par catégorie"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    limit = request.args.get('limit', 8, type=int)
    
    # Calculer les métriques par catégorie
    category_stats = df_companies.groupby('Main Category').agg({
        'Rating': 'mean',
        'Score': 'mean',
        'Reviews': 'mean',
        'Name': 'count'
    }).rename(columns={'Name': 'Count'})
    
    # Sélectionner les top catégories par nombre d'entreprises
    top_categories = category_stats.nlargest(limit, 'Count')
    
    # Normaliser les valeurs (0-100)
    normalized_data = []
    for category, stats in top_categories.iterrows():
        normalized_data.append({
            'category': category,
            'rating': round((stats['Rating'] / 5) * 100, 1) if pd.notna(stats['Rating']) else 0,
            'score': round(stats['Score'], 1) if pd.notna(stats['Score']) else 0,
            'popularity': min(round((stats['Reviews'] / top_categories['Reviews'].max()) * 100, 1), 100) if pd.notna(stats['Reviews']) else 0,
            'presence': round((stats['Count'] / top_categories['Count'].max()) * 100, 1)
        })
    
    result = {
        'chart_type': 'radar',
        'title': f'Performance Multi-Critères par Catégorie (Top {limit})',
        'data': {
            'categories': [item['category'] for item in normalized_data],
            'metrics': ['Rating', 'Score', 'Popularité', 'Présence'],
            'values': [
                [item['rating'], item['score'], item['popularity'], item['presence']]
                for item in normalized_data
            ]
        }
    }
    
    return jsonify(result)

@app.route('/api/charts/geographic-heatmap', methods=['GET'])
def geographic_heatmap():
    """Données pour heatmap géographique"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    # Filtrer les données avec coordonnées
    geo_data = df_companies[(df_companies['Latitude'].notna()) & (df_companies['Longitude'].notna())]
    
    # Grouper par zones géographiques approximatives
    geo_data_sample = geo_data.sample(min(500, len(geo_data)))  # Limiter pour les performances
    
    heatmap_points = []
    for _, row in geo_data_sample.iterrows():
        heatmap_points.append({
            'lat': float(row['Latitude']),
            'lng': float(row['Longitude']),
            'weight': float(row.get('Score', 50)),  # Utiliser le score comme poids
            'city': row['City'],
            'name': row['Name']
        })
    
    result = {
        'chart_type': 'heatmap',
        'title': 'Répartition Géographique des Entreprises',
        'data': {
            'points': heatmap_points,
            'center': {
                'lat': geo_data['Latitude'].mean(),
                'lng': geo_data['Longitude'].mean()
            }
        },
        'total_points': len(heatmap_points)
    }
    
    return jsonify(result)

# ==================== DASHBOARD DATA ====================

@app.route('/api/charts/dashboard-summary', methods=['GET'])
def dashboard_summary():
    """Données résumées pour dashboard"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    # KPIs principaux
    total_companies = len(df_companies)
    avg_rating = df_companies['Rating'].mean()
    avg_score = df_companies['Score'].mean()
    total_reviews = df_companies['Reviews'].sum()
    
    # Top 5 de chaque catégorie
    top_cities = df_companies['City'].value_counts().head(5).to_dict()
    top_categories = df_companies['Main Category'].value_counts().head(5).to_dict()
    score_distribution = df_companies['Score Category'].value_counts().to_dict()
    
    result = {
        'kpis': {
            'total_companies': int(total_companies),
            'avg_rating': round(float(avg_rating), 2) if pd.notna(avg_rating) else None,
            'avg_score': round(float(avg_score), 2) if pd.notna(avg_score) else None,
            'total_reviews': int(total_reviews) if pd.notna(total_reviews) else 0,
            'unique_cities': int(df_companies['City'].nunique()),
            'unique_categories': int(df_companies['Main Category'].nunique())
        },
        'top_data': {
            'cities': top_cities,
            'categories': top_categories,
            'score_distribution': score_distribution
        },
        'data_quality': {
            'companies_with_rating': int(df_companies['Rating'].notna().sum()),
            'companies_with_score': int(df_companies['Score'].notna().sum()),
            'companies_with_reviews': int(df_companies['Reviews'].notna().sum()),
            'companies_with_website': int(df_companies['Website'].notna().sum())
        }
    }
    
    return jsonify(result)


if __name__ == '__main__':
    # Essayer de charger les données au démarrage si le fichier existe
    if os.path.exists('uploads/companies_data.csv'):
        load_csv_data()
    
    app.run(debug=True, host='0.0.0.0', port=5001)