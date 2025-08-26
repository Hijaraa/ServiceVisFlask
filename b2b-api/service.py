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

# Mettre à jour la route home() pour inclure les nouveaux endpoints
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Service Flask - Statistiques pour Charts B2B",
        "version": "1.1",
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
            },
            "new_creative_charts": {
                "business_maturity_bubble": "GET /api/charts/business-maturity-bubble",
                "market_share_donut": "GET /api/charts/market-share-donut?limit=8",
                "performance_matrix": "GET /api/charts/performance-matrix?city_limit=10&category_limit=8",
                "growth_potential_funnel": "GET /api/charts/growth-potential-funnel",
                "competitive_landscape_treemap": "GET /api/charts/competitive-landscape-treemap",
                "market_evolution_area": "GET /api/charts/market-evolution-area",
                "quality_vs_popularity_quadrant": "GET /api/charts/quality-vs-popularity-quadrant"
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
# ==================== ADDITIONAL CREATIVE CHARTS ====================

@app.route('/api/charts/business-maturity-bubble', methods=['GET'])
def business_maturity_bubble():
    """Données pour bubble chart - Maturité des entreprises (Rating vs Reviews vs Score)"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    # Filtrer les données complètes
    bubble_data = df_companies[
        (df_companies['Rating'].notna()) & 
        (df_companies['Reviews'].notna()) & 
        (df_companies['Score'].notna())
    ]
    
    # Limiter pour les performances
    if len(bubble_data) > 300:
        bubble_data = bubble_data.sample(300)
    
    bubbles = []
    for _, row in bubble_data.iterrows():
        bubbles.append({
            'x': float(row['Rating']),  # Rating sur X
            'y': float(row['Score']),   # Score sur Y
            'size': min(int(row['Reviews']) / 10, 100),  # Taille basée sur Reviews
            'name': row['Name'][:25] + '...' if len(row['Name']) > 25 else row['Name'],
            'city': row['City'],
            'category': row['Main Category'],
            'reviews': int(row['Reviews'])
        })
    
    result = {
        'chart_type': 'bubble',
        'title': 'Maturité des Entreprises (Rating vs Score vs Popularité)',
        'data': {
            'bubbles': bubbles
        },
        'axis_labels': {
            'x': 'Rating (1-5)',
            'y': 'Score (0-100)',
            'size': 'Nombre de Reviews'
        },
        'total_companies': len(bubbles)
    }
    
    return jsonify(result)

@app.route('/api/charts/market-share-donut', methods=['GET'])
def market_share_donut():
    """Données pour donut chart - Parts de marché par catégorie avec détails"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    limit = request.args.get('limit', 8, type=int)
    
    # Calculer la part de marché avec moyenne des scores
    market_data = df_companies.groupby('Main Category').agg({
        'Name': 'count',
        'Score': 'mean',
        'Rating': 'mean',
        'Reviews': 'sum'
    }).rename(columns={'Name': 'Count'})
    
    top_categories = market_data.nlargest(limit, 'Count')
    total_companies = len(df_companies)
    others_count = total_companies - int(top_categories['Count'].sum())
    
    # Préparer les données avec métriques enrichies
    segments = []
    for category, stats in top_categories.iterrows():
        segments.append({
            'label': category,
            'value': int(stats['Count']),
            'percentage': round((stats['Count'] / total_companies) * 100, 1),
            'avg_score': round(stats['Score'], 1) if pd.notna(stats['Score']) else 'N/A',
            'avg_rating': round(stats['Rating'], 2) if pd.notna(stats['Rating']) else 'N/A',
            'total_reviews': int(stats['Reviews']) if pd.notna(stats['Reviews']) else 0
        })
    
    if others_count > 0:
        segments.append({
            'label': 'Autres catégories',
            'value': others_count,
            'percentage': round((others_count / total_companies) * 100, 1),
            'avg_score': 'N/A',
            'avg_rating': 'N/A',
            'total_reviews': 0
        })
    
    result = {
        'chart_type': 'donut',
        'title': f'Parts de Marché par Catégorie (Top {limit})',
        'data': {
            'segments': segments,
            'total': total_companies,
            'center_text': f'{total_companies} Entreprises'
        }
    }
    
    return jsonify(result)

@app.route('/api/charts/performance-matrix', methods=['GET'])
def performance_matrix():
    """Données pour matrix chart - Matrice Performance par Ville/Catégorie"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    city_limit = request.args.get('city_limit', 10, type=int)
    category_limit = request.args.get('category_limit', 8, type=int)
    
    # Top villes et catégories
    top_cities = df_companies['City'].value_counts().head(city_limit).index.tolist()
    top_categories = df_companies['Main Category'].value_counts().head(category_limit).index.tolist()
    
    # Créer la matrice
    matrix_data = []
    for i, city in enumerate(top_cities):
        for j, category in enumerate(top_categories):
            subset = df_companies[
                (df_companies['City'] == city) & 
                (df_companies['Main Category'] == category)
            ]
            
            if len(subset) > 0:
                avg_score = subset['Score'].mean()
                company_count = len(subset)
                avg_rating = subset['Rating'].mean()
                
                matrix_data.append({
                    'x': j,  # Position catégorie
                    'y': i,  # Position ville
                    'value': round(avg_score, 1) if pd.notna(avg_score) else 0,
                    'count': company_count,
                    'avg_rating': round(avg_rating, 2) if pd.notna(avg_rating) else 0,
                    'city': city,
                    'category': category
                })
    
    result = {
        'chart_type': 'matrix',
        'title': 'Matrice Performance Score par Ville/Catégorie',
        'data': {
            'matrix': matrix_data,
            'x_labels': top_categories,  # Catégories
            'y_labels': top_cities,     # Villes
        },
        'axis_labels': {
            'x': 'Catégories',
            'y': 'Villes'
        }
    }
    
    return jsonify(result)

@app.route('/api/charts/growth-potential-funnel', methods=['GET'])
def growth_potential_funnel():
    """Données pour funnel chart - Potentiel de croissance par niveau de score"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    # Définir les niveaux de potentiel
    score_ranges = [
        {'name': 'Excellence (80-100)', 'min': 80, 'max': 100, 'color': '#28a745'},
        {'name': 'Très Bon (60-79)', 'min': 60, 'max': 79, 'color': '#17a2b8'},
        {'name': 'Bon (40-59)', 'min': 40, 'max': 59, 'color': '#ffc107'},
        {'name': 'Moyen (20-39)', 'min': 20, 'max': 39, 'color': '#fd7e14'},
        {'name': 'Faible (0-19)', 'min': 0, 'max': 19, 'color': '#dc3545'}
    ]
    
    funnel_data = []
    companies_with_score = df_companies[df_companies['Score'].notna()]
    
    for level in score_ranges:
        subset = companies_with_score[
            (companies_with_score['Score'] >= level['min']) & 
            (companies_with_score['Score'] <= level['max'])
        ]
        
        if len(subset) > 0:
            avg_rating = subset['Rating'].mean()
            avg_reviews = subset['Reviews'].mean()
            
            funnel_data.append({
                'label': level['name'],
                'value': len(subset),
                'percentage': round((len(subset) / len(companies_with_score)) * 100, 1),
                'avg_rating': round(avg_rating, 2) if pd.notna(avg_rating) else 'N/A',
                'avg_reviews': round(avg_reviews, 0) if pd.notna(avg_reviews) else 0,
                'color': level['color']
            })
    
    result = {
        'chart_type': 'funnel',
        'title': 'Distribution du Potentiel de Croissance',
        'data': {
            'stages': funnel_data,
            'total_companies': len(companies_with_score)
        }
    }
    
    return jsonify(result)

@app.route('/api/charts/competitive-landscape-treemap', methods=['GET'])
def competitive_landscape_treemap():
    """Données pour treemap chart - Paysage concurrentiel hiérarchique"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    # Grouper par catégorie puis par ville
    treemap_data = []
    
    categories = df_companies['Main Category'].value_counts().head(6)
    
    for category, total_count in categories.items():
        category_data = df_companies[df_companies['Main Category'] == category]
        cities_in_category = category_data['City'].value_counts().head(5)
        
        children = []
        for city, city_count in cities_in_category.items():
            city_subset = category_data[category_data['City'] == city]
            avg_score = city_subset['Score'].mean()
            avg_rating = city_subset['Rating'].mean()
            
            children.append({
                'name': f"{city} ({city_count})",
                'value': city_count,
                'avg_score': round(avg_score, 1) if pd.notna(avg_score) else 0,
                'avg_rating': round(avg_rating, 2) if pd.notna(avg_rating) else 0,
                'color_intensity': avg_score if pd.notna(avg_score) else 50
            })
        
        # Ajouter "Autres villes" si nécessaire
        others_count = total_count - sum([child['value'] for child in children])
        if others_count > 0:
            children.append({
                'name': f"Autres villes ({others_count})",
                'value': others_count,
                'avg_score': 0,
                'avg_rating': 0,
                'color_intensity': 30
            })
        
        treemap_data.append({
            'name': category,
            'value': total_count,
            'children': children
        })
    
    result = {
        'chart_type': 'treemap',
        'title': 'Paysage Concurrentiel par Catégorie et Ville',
        'data': {
            'tree': treemap_data
        }
    }
    
    return jsonify(result)

@app.route('/api/charts/market-evolution-area', methods=['GET'])
def market_evolution_area():
    """Données pour area chart - Évolution du marché par catégorie de score"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    # Simuler une évolution temporelle basée sur les données
    # (Dans un vrai cas, vous auriez des données historiques)
    
    categories = ['Excellent', 'Très Bon', 'Bon', 'Moyen', 'Faible']
    score_distribution = df_companies['Score Category'].value_counts()
    
    # Créer des données d'évolution simulées (12 mois)
    months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
              'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
    
    evolution_data = []
    
    for category in categories:
        if category in score_distribution:
            base_value = score_distribution[category]
            # Simuler des variations mensuelles
            monthly_values = []
            for i in range(12):
                # Variation simulée de ±15%
                variation = np.random.uniform(0.85, 1.15)
                monthly_values.append(int(base_value * variation))
            
            evolution_data.append({
                'name': category,
                'data': monthly_values
            })
    
    result = {
        'chart_type': 'area',
        'title': 'Évolution Simulée du Marché par Performance',
        'data': {
            'categories': months,
            'series': evolution_data
        },
        'note': 'Données simulées à des fins de démonstration'
    }
    
    return jsonify(result)

@app.route('/api/charts/quality-vs-popularity-quadrant', methods=['GET'])
def quality_vs_popularity_quadrant():
    """Données pour quadrant chart - Qualité vs Popularité"""
    if df_companies is None:
        return jsonify({'error': 'Aucune donnée chargée'}), 400
    
    # Filtrer les données complètes
    quadrant_data = df_companies[
        (df_companies['Score'].notna()) & 
        (df_companies['Reviews'].notna())
    ]
    
    if len(quadrant_data) == 0:
        return jsonify({'error': 'Pas assez de données pour le quadrant'}), 400
    
    # Calculer les médianes pour diviser en quadrants
    median_score = quadrant_data['Score'].median()
    median_reviews = quadrant_data['Reviews'].median()
    
    # Catégoriser les entreprises
    quadrants = {
        'stars': [],      # Haute qualité, Haute popularité
        'hidden_gems': [], # Haute qualité, Faible popularité  
        'popular': [],    # Faible qualité, Haute popularité
        'laggards': []    # Faible qualité, Faible popularité
    }
    
    for _, row in quadrant_data.iterrows():
        score = row['Score']
        reviews = row['Reviews']
        
        point = {
            'name': row['Name'][:20] + '...' if len(row['Name']) > 20 else row['Name'],
            'score': float(score),
            'reviews': int(reviews),
            'city': row['City'],
            'category': row['Main Category'],
            'rating': float(row['Rating']) if pd.notna(row['Rating']) else None
        }
        
        if score >= median_score and reviews >= median_reviews:
            quadrants['stars'].append(point)
        elif score >= median_score and reviews < median_reviews:
            quadrants['hidden_gems'].append(point)
        elif score < median_score and reviews >= median_reviews:
            quadrants['popular'].append(point)
        else:
            quadrants['laggards'].append(point)
    
    result = {
        'chart_type': 'quadrant',
        'title': 'Matrice Qualité vs Popularité',
        'data': {
            'quadrants': quadrants,
            'medians': {
                'score': float(median_score),
                'reviews': float(median_reviews)
            },
            'labels': {
                'stars': 'Étoiles (Qualité+, Pop+)',
                'hidden_gems': 'Perles Cachées (Qualité+, Pop-)',
                'popular': 'Populaires (Qualité-, Pop+)',
                'laggards': 'Retardataires (Qualité-, Pop-)'
            }
        },
        'axis_labels': {
            'x': 'Nombre de Reviews (Popularité)',
            'y': 'Score (Qualité)'
        },
        'counts': {
            'stars': len(quadrants['stars']),
            'hidden_gems': len(quadrants['hidden_gems']),
            'popular': len(quadrants['popular']),
            'laggards': len(quadrants['laggards'])
        }
    }
    
    return jsonify(result)

if __name__ == '__main__':
    # Essayer de charger les données au démarrage si le fichier existe
    if os.path.exists('uploads/companies_data.csv'):
        load_csv_data()
    
    app.run(debug=True, host='0.0.0.0', port=5001)