# app.py (Backend)
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from io import StringIO

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "http://localhost:*"}})

# Hardcoded B2B and B2C data from dashboards
b2b_csv = """name,primary_type,city,rating,reviews,address,phone,description,score,quality,website
Enterprise 1,Agence événementielle,Other,4.5,816,Address in Other,0123456789,Description for Agence événementielle,96.50107558790097,Excellent,
Enterprise 2,Agence événementielle,Other,4.4,613,Address in Other,0123456789,Description for Agence événementielle,91.5906779037661,Average,www.example.com
Enterprise 3,Agence événementielle,Other,4.9,697,Address in Other,0123456789,Description for Agence événementielle,87.3528703385734,Good,www.example.com
Enterprise 4,Agence événementielle,Other,0.6,674,Address in Other,0123456789,Description for Agence événementielle,79.35756400822143,Average,www.example.com
Enterprise 5,Agence événementielle,Other,4.1,713,Address in Other,0123456789,Description for Agence événementielle,86.94741959359804,Average,www.example.com
Enterprise 6,Agence événementielle,Other,4.0,968,Address in Other,0123456789,Description for Agence événementielle,61.954998961302366,Excellent,www.example.com
Enterprise 7,Agence événementielle,Other,4.5,355,Address in Other,0123456789,Description for Agence événementielle,51.68614995393899,Good,www.example.com
Enterprise 8,Agence événementielle,Lyon,4.3,592,Address in Lyon,0123456789,Description for Agence événementielle,57.87930269392635,Good,
Enterprise 9,Agence événementielle,Other,4.6,977,Address in Other,0123456789,Description for Agence événementielle,62.95368706406923,Average,www.example.com
Enterprise 10,Agence événementielle,Other,0.1,552,Address in Other,0123456789,Description for Agence événementielle,68.08079795951596,Average,www.example.com
Enterprise 11,Agence événementielle,Other,4.4,417,Address in Other,0123456789,Description for Agence événementielle,75.18460997393993,Excellent,www.example.com
Enterprise 12,Agence événementielle,Other,4.7,805,Address in Other,0123456789,Description for Agence événementielle,73.83895111614184,Excellent,www.example.com
Enterprise 13,Agence événementielle,Other,4.6,945,Address in Other,0123456789,Description for Agence événementielle,72.9436922945515,Excellent,www.example.com
Enterprise 14,Agence événementielle,Other,4.4,412,Address in Other,0123456789,Description for Agence événementielle,77.80244013005625,Excellent,www.example.com
Enterprise 15,Agence événementielle,Other,4.9,534,Address in Other,0123456789,Description for Agence événementielle,65.63585643908951,Excellent,www.example.com
Enterprise 16,Agence événementielle,Other,4.6,485,Address in Other,0123456789,Description for Agence événementielle,64.02844507676339,Bad,
Enterprise 17,Agence événementielle,Other,4.5,73,Address in Other,0123456789,Description for Agence événementielle,73.83895111614184,Average,www.example.com
Enterprise 18,Agence événementielle,Other,5.0,986,Address in Other,0123456789,Description for Agence événementielle,77.91773428882085,Good,
Enterprise 19,Agence événementielle,Other,4.4,714,Address in Other,0123456789,Description for Agence événementielle,48.3849573325759,Excellent,
Enterprise 20,Agence événementielle,Other,4.6,455,Address in Other,0123456789,Description for Agence événementielle,80.05395721416544,Average,
Enterprise 21,Organisateur d'événements,Other,4.6,928,Address in Other,0123456789,Description for Organisateur d'événements,72.0660374334522,Excellent,www.example.com
Enterprise 22,Organisateur d'événements,Other,4.7,814,Address in Other,0123456789,Description for Organisateur d'événements,70.01632559541072,Good,www.example.com
Enterprise 23,Organisateur d'événements,Other,0.9,453,Address in Other,0123456789,Description for Organisateur d'événements,67.44985581485234,Good,www.example.com
Enterprise 24,Organisateur d'événements,Other,4.9,568,Address in Other,0123456789,Description for Organisateur d'événements,98.48995569551747,Excellent,
Enterprise 25,Organisateur d'événements,Other,3.7,119,Address in Other,0123456789,Description for Organisateur d'événements,76.7391852937802,Excellent,www.example.com
Enterprise 26,Organisateur d'événements,Other,4.3,875,Address in Other,0123456789,Description for Organisateur d'événements,70.8212840828434,Average,www.example.com
Enterprise 27,Organisateur d'événements,Other,4.9,590,Address in Other,0123456789,Description for Organisateur d'événements,82.0959825035664,Excellent,www.example.com
Enterprise 28,Organisateur d'événements,Other,0.0,334,Address in Other,0123456789,Description for Organisateur d'événements,89.76246910902705,Excellent,www.example.com
Enterprise 29,Organisateur d'événements,Other,4.1,165,Address in Other,0123456789,Description for Organisateur d'événements,80.05966521336384,Excellent,www.example.com
Enterprise 30,Mobile phone repair shop,Other,4.8,96,Address in Other,0123456789,Description for Mobile phone repair shop,73.67329992796233,Average,www.example.com
Enterprise 31,Mobile phone repair shop,Other,4.6,981,Address in Other,0123456789,Description for Mobile phone repair shop,73.99428838619203,Excellent,www.example.com
Enterprise 32,Mobile phone repair shop,Other,4.8,939,Address in Other,0123456789,Description for Mobile phone repair shop,72.70116246347207,Excellent,www.example.com
Enterprise 33,Mobile phone repair shop,Marseille,4.9,122,Address in Marseille,0123456789,Description for Mobile phone repair shop,64.87802057333218,Good,www.example.com
Enterprise 34,Cell phone store,Other,4.5,343,Address in Other,0123456789,Description for Cell phone store,67.48559686483846,Good,www.example.com
Enterprise 35,Cell phone store,Toulouse,4.2,199,Address in Toulouse,0123456789,Description for Cell phone store,59.96370450585025,Good,www.example.com
Enterprise 36,Cell phone store,Paris,4.2,231,Address in Paris,0123456789,Description for Cell phone store,87.06843822493224,Good,
Enterprise 37,Cell phone store,Other,0.7,721,Address in Other,0123456789,Description for Cell phone store,81.25867080039778,Excellent,
Enterprise 38,Phone repair service,Other,4.3,980,Address in Other,0123456789,Description for Phone repair service,67.97273861539415,Excellent,www.example.com
Enterprise 39,Phone repair service,Other,4.6,519,Address in Other,0123456789,Description for Phone repair service,84.09518143982854,Good,
Enterprise 40,Phone repair service,Other,1.0,560,Address in Other,0123456789,Description for Phone repair service,70.48865512282204,Average,www.example.com
Enterprise 41,Other,Other,4.2,175,Address in Other,0123456789,Description for Other,93.43386718019681,Good,www.example.com
Enterprise 42,Other,Nice,0.8,587,Address in Nice,0123456789,Description for Other,77.39615351929102,Average,
Enterprise 43,Other,Other,4.5,360,Address in Other,0123456789,Description for Other,62.900370924005934,Excellent,
Enterprise 44,Other,Marseille,4.8,599,Address in Marseille,0123456789,Description for Other,84.8912835247048,Average,www.example.com
Enterprise 45,Other,Other,0.4,477,Address in Other,0123456789,Description for Other,50.80095485926451,Excellent,
Enterprise 46,Other,Paris,0.0,43,Address in Paris,0123456789,Description for Other,75.29531241141126,Average,www.example.com
Enterprise 47,Other,Other,4.2,799,Address in Other,0123456789,Description for Other,68.50280860654529,Average,www.example.com
Enterprise 48,Other,Other,0.4,652,Address in Other,0123456789,Description for Other,76.58030075101439,Excellent,www.example.com
Enterprise 49,Other,Other,4.1,122,Address in Other,0123456789,Description for Other,67.47337020072362,Good,
Enterprise 50,Other,Other,4.9,852,Address in Other,0123456789,Description for Other,65.47785539237098,Excellent,www.example.com
Enterprise 51,Other,Other,4.0,747,Address in Other,0123456789,Description for Other,75.64676395881781,Good,www.example.com
Enterprise 52,Other,Other,4.0,265,Address in Other,0123456789,Description for Other,82.55125554444855,Average,www.example.com
Enterprise 53,Other,Other,4.6,363,Address in Other,0123456789,Description for Other,71.08846869014218,Good,www.example.com
Enterprise 54,Other,Other,4.7,805,Address in Other,0123456789,Description for Other,85.82914572389069,Excellent,www.example.com
Enterprise 55,Other,Other,4.3,606,Address in Other,0123456789,Description for Other,88.4146578903983,Average,www.example.com
Enterprise 56,Other,Other,4.6,177,Address in Other,0123456789,Description for Other,64.42854661676076,Good,www.example.com
Enterprise 57,Other,Other,4.1,959,Address in Other,0123456789,Description for Other,77.51339814622568,Excellent,www.example.com
Enterprise 58,Other,Other,4.6,224,Address in Other,0123456789,Description for Other,59.17892870233735,Excellent,www.example.com
Enterprise 59,Other,Paris,4.5,903,Address in Paris,0123456789,Description for Other,88.65301075654568,Excellent,www.example.com
Enterprise 60,Other,Other,4.5,74,Address in Other,0123456789,Description for Other,62.02370839263429,Average,www.example.com
Enterprise 61,Other,Other,4.3,186,Address in Other,0123456789,Description for Other,65.1822869096967,Excellent,www.example.com
Enterprise 62,Other,Other,4.8,34,Address in Other,0123456789,Description for Other,76.5911941176649,Average,www.example.com
Enterprise 63,Other,Other,4.4,621,Address in Other,0123456789,Description for Other,87.2454686231262,Excellent,www.example.com
Enterprise 64,Other,Other,4.5,309,Address in Other,0123456789,Description for Other,72.9436922945515,Good,www.example.com
Enterprise 65,Other,Other,4.2,799,Address in Other,0123456789,Description for Other,82.16271937710101,Good,www.example.com
Enterprise 66,Other,Other,4.4,239,Address in Other,0123456789,Description for Other,95.05734041751853,Good,www.example.com
Enterprise 67,Other,Other,4.3,75,Address in Other,0123456789,Description for Other,66.66707333611525,Excellent,www.example.com
Enterprise 68,Other,Other,4.3,556,Address in Other,0123456789,Description for Other,70.51786843890297,Good,
Enterprise 69,Other,Other,4.9,289,Address in Other,0123456789,Description for Other,80.6352864085961,Average,www.example.com
Enterprise 70,Other,Other,3.1,286,Address in Other,0123456789,Description for Other,63.12219294645083,Good,www.example.com
Enterprise 71,Other,Other,3.5,984,Address in Other,0123456789,Description for Other,63.12219294645083,Good,www.example.com
Enterprise 72,Other,Other,4.9,747,Address in Other,0123456789,Description for Other,72.9436922945515,Average,www.example.com
Enterprise 73,Other,Other,4.2,799,Address in Other,0123456789,Description for Other,85.82914572389069,Average,www.example.com
Enterprise 74,Other,Other,4.7,23,Address in Other,0123456789,Description for Other,76.39071159427502,Excellent,
Enterprise 75,Other,Other,4.7,819,Address in Other,0123456789,Description for Other,67.26162646521264,Average,www.example.com
Enterprise 76,Other,Other,4.9,467,Address in Other,0123456789,Description for Other,62.669685099949575,Good,www.example.com
Enterprise 77,Other,Other,4.5,172,Address in Other,0123456789,Description for Other,63.13067025538903,Excellent,www.example.com
Enterprise 78,Other,Other,2.0,192,Address in Other,0123456789,Description for Other,69.03541102202834,Excellent,www.example.com
Enterprise 79,Other,Other,0.2,657,Address in Other,0123456789,Description for Other,80.96376683026935,Average,www.example.com
Enterprise 80,Other,Other,4.3,600,Address in Other,0123456789,Description for Other,69.81671152797234,Excellent,
Enterprise 81,Other,Other,4.4,225,Address in Other,0123456789,Description for Other,61.99517128891239,Good,
Enterprise 82,Other,Toulouse,4.7,422,Address in Toulouse,0123456789,Description for Other,74.11315066503155,Excellent,www.example.com
Enterprise 83,Other,Other,4.7,855,Address in Other,0123456789,Description for Other,80.7669247130217,Excellent,www.example.com
Enterprise 84,Other,Other,4.6,608,Address in Other,0123456789,Description for Other,93.16621953136273,Excellent,
Enterprise 85,Other,Toulouse,0.2,198,Address in Toulouse,0123456789,Description for Other,63.06265160318685,Good,www.example.com
Enterprise 86,Other,Other,4.6,955,Address in Other,0123456789,Description for Other,68.45510145236995,Good,www.example.com
Enterprise 87,Other,Other,0.6,782,Address in Other,0123456789,Description for Other,84.30692450799287,Average,www.example.com
Enterprise 88,Other,Other,3.4,6,Address in Other,0123456789,Description for Other,83.69823954402328,Excellent,www.example.com
Enterprise 89,Other,Other,5.0,729,Address in Other,0123456789,Description for Other,62.3751977043401,Excellent,www.example.com
Enterprise 90,Other,Other,5.0,960,Address in Other,0123456789,Description for Other,72.9436922945515,Excellent,www.example.com
Enterprise 91,Other,Other,0.7,924,Address in Other,0123456789,Description for Other,95.21061042235466,Average,www.example.com
Enterprise 92,Other,Other,4.1,615,Address in Other,0123456789,Description for Other,73.06999889635235,Average,
Enterprise 93,Other,Other,4.4,24,Address in Other,0123456789,Description for Other,71.19569986934859,Average,www.example.com
Enterprise 94,Other,Other,4.3,3,Address in Other,0123456789,Description for Other,90.16355738416789,Excellent,www.example.com
Enterprise 95,Other,Other,4.1,987,Address in Other,0123456789,Description for Other,73.07007632104819,Average,www.example.com
Enterprise 96,Other,Other,4.9,41,Address in Other,0123456789,Description for Other,67.4684949371562,Excellent,www.example.com
Enterprise 97,Other,Other,3.0,852,Address in Other,0123456789,Description for Other,60.26442829890122,Good,www.example.com
Enterprise 98,Other,Other,4.5,798,Address in Other,0123456789,Description for Other,55.577438327401566,Good,www.example.com
Enterprise 99,Other,Other,0.6,460,Address in Other,0123456789,Description for Other,64.09364175202202,Good,
Enterprise 100,Other,Other,4.6,725,Address in Other,0123456789,Description for Other,66.96867774127296,Good,www.example.com
"""

b2c_csv = """name,gender,age,employment,satisfaction,city,relationship_status,origin_city,phone
Client 1,female,62,,3.0,Paris,Non spécifié,Other,0123456789
Client 2,male,23,,1.6,Paris,Non spécifié,Paris,0123456789
Client 3,male,24,gucci,5.0,Paris,Non spécifié,Paris,0123456789
Client 4,male,34,,3.6,Paris,Non spécifié,Paris,0123456789
Client 5,female,19,,2.3,Paris,Non spécifié,Paris,0123456789
Client 6,male,37,cabinet dentaire orio,2.5,Paris,Non spécifié,Other,0123456789
Client 7,male,48,self-employed,3.0,Paris,Non spécifié,Other,0123456789
Client 8,female,41,,2.2,Paris,Non spécifié,Other,0123456789
Client 9,female,28,,4.4,Paris,single,,0123456789
Client 10,female,24,,1.7,Paris,single,Paris,0123456789
Client 11,female,36,,4.4,Paris,Non spécifié,Other,0123456789
Client 12,male,52,,2.9,Paris,Non spécifié,Paris,0123456789
Client 13,female,63,,4.1,Paris,Non spécifié,Paris,0123456789
Client 14,male,55,,1.6,Paris,Non spécifié,Other,0123456789
Client 15,female,22,self-employed,3.2,Paris,married,Other,0123456789
Client 16,male,26,,3.8,Paris,Non spécifié,Paris,0123456789
Client 17,male,30,self-employed,3.4,Paris,Non spécifié,,0123456789
Client 18,male,24,,4.5,Paris,Non spécifié,Paris,0123456789
Client 19,inconnu,18,,3.8,Paris,Non spécifié,Other,0123456789
Client 20,male,50,,4.5,Paris,Non spécifié,,0123456789
Client 21,inconnu,19,,3.4,Paris,Non spécifié,Other,0123456789
Client 22,male,37,,3.5,Paris,Non spécifié,Paris,0123456789
Client 23,female,36,,2.6,Paris,single,Paris,0123456789
Client 24,male,64,,2.7,Paris,Non spécifié,Paris,0123456789
Client 25,female,63,,1.7,Paris,Non spécifié,Other,0123456789
Client 26,male,19,,3.0,Paris,Non spécifié,,0123456789
Client 27,male,29,,1.9,Paris,Non spécifié,Paris,0123456789
Client 28,male,35,,4.3,Paris,Non spécifié,Other,0123456789
Client 29,male,64,passage porte & polignes,3.1,Paris,single,,0123456789
Client 30,male,46,self-employed,3.5,Paris,Non spécifié,,0123456789
Client 31,inconnu,59,,4.4,Paris,Non spécifié,,0123456789
Client 32,male,50,,3.6,Paris,in a relationship,,0123456789
Client 33,male,66,passage porte & polignes,3.8,Paris,Non spécifié,Paris,0123456789
Client 34,male,56,self-employed,4.6,Paris,Non spécifié,Other,0123456789
Client 35,male,32,,4.7,Paris,Non spécifié,Paris,0123456789
Client 36,male,55,,4.5,Paris,Non spécifié,Other,0123456789
Client 37,male,47,,2.6,Paris,married,Other,0123456789
Client 38,male,27,,1.5,Paris,Non spécifié,Paris,0123456789
Client 39,male,55,parasitic tour,1.1,Paris,Non spécifié,Other,0123456789
Client 40,male,69,,4.6,Paris,in a relationship,Paris,0123456789
"""

# Load  data
dataframes = {
    'b2b': pd.read_csv(StringIO(b2b_csv)),
    'b2c': pd.read_csv(StringIO(b2c_csv))
}

# Clean data
for mode in ['b2b', 'b2c']:
    df = dataframes[mode]
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    if 'satisfaction' in df.columns:
        df['satisfaction'] = pd.to_numeric(df['satisfaction'], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), None)

def clean_for_json(data):
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_json(i) for i in data]
    elif isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
        return None
    elif pd.isna(data):
        return None
    else:
        return data

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '').strip().lower()
    mode = data.get('mode', '').lower()

    if not query:
        return jsonify({'error': 'Question vide.'}), 400
    if mode not in ['b2b', 'b2c']:
        return jsonify({'error': 'Mode invalide.'}), 400
    df = dataframes.get(mode)
    try:
        response, chart_data = analyze_query(df, query, mode)
        return jsonify({
            'response': response,
            'update_data': clean_for_json(chart_data) if chart_data else None
        })
    except Exception as e:
        return jsonify({'error': f'Erreur lors de l\'analyse : {str(e)}'}), 500

def analyze_query(df, query, mode):
    # Common queries
    if any(q in query for q in ['total', 'combien', 'nombre', 'total enterprises', 'total clients']):
        if mode == 'b2b':
            return f"📊 Il y a **{len(df)} entreprises** dans la base B2B.", {
                'chart_type': 'bar',
                'title': 'Nombre total d’entreprises',
                'labels': ['Total'],
                'values': [len(df)]
            }
        else:
            return f"📊 Il y a **{len(df)} clients** dans la base B2C.", {
                'chart_type': 'bar',
                'title': 'Nombre total de clients',
                'labels': ['Total'],
                'values': [len(df)]
            }

    if 'type' in query or 'activité' in query or 'catégories' in query or 'distribution par catégories' in query:
        if mode == 'b2b' and 'primary_type' in df.columns:
            counts = df['primary_type'].value_count()
            top = counts.index[0] if len(counts) > 0 else "Inconnu"
            return f"🔍 **Distribution par catégories :** {top} est le plus courant ({counts.iloc[0]}). Total types uniques : **{len(counts)}**.\nTop 5 :\n" + "\n".join([f"• {t} ({c})" for t, c in counts.head(5).items()]), {
                'chart_type': 'pie',
                'title': 'Distribution par Catégories',
                'labels': counts.index.tolist(),
                'values': counts.values.tolist()
            }

    if any(q in query for q in ['note', 'rating', 'satisfaction', 'rating moyen']):
        if 'rating' in df.columns:
            avg = df['rating'].mean()
            return f"⭐ **Note moyenne :** **{avg:.2f}/5**.", {
                'chart_type': 'bar',
                'title': 'Note moyenne',
                'labels': ['Moyenne'],
                'values': [round(avg, 2)]
            }

    if 'ville' in query or 'localité' in query or 'top villes' in query:
        if 'city' in df.columns:
            city_counts = df['city'].value_counts().dropna().head(10)
            top_city = city_counts.index[0] if len(city_counts) > 0 else "Inconnue"
            return f"🌆 **Répartition par ville :** Ville la plus représentée : **{top_city}** ({city_counts.iloc[0]}).\nNombre de villes : **{len(city_counts)}**.\nTop 5 :\n" + "\n".join([f"• {c} : {n}" for c, n in city_counts.head(5).items()]), {
                'chart_type': 'bar',
                'title': 'Entreprises/Clients par ville',
                'labels': city_counts.index.tolist(),
                'values': city_counts.values.tolist()
            }

    if 'qualité' in query or 'quality' in query or 'qualité des entreprises':
        if mode == 'b2b' and 'quality' in df.columns:
            counts = df['quality'].value_counts().dropna()
            return f"📊 **Qualité des entreprises :** Excellent {counts.get('Excellent', 0)}%, Good {counts.get('Good', 0)}%, etc.", {
                'chart_type': 'pie',
                'title': 'Qualité des Entreprises',
                'labels': counts.index.tolist(),
                'values': counts.values.tolist()
            }

    if 'potentiel de croissance' in query or 'growth potential' in query:
        if mode == 'b2b':
            # Hardcode from dashboard, or calculate if data allows
            # Assume from quality or score
            high_growth = len(df[df['score'] > 80]) if 'score' in df.columns else 0
            return f"📈 **Potentiel de croissance :** Excellent : {high_growth} ({round(high_growth / len(df) * 100, 1)}%).", {
                'chart_type': 'bar',
                'title': 'Potentiel de Croissance',
                'labels': ['Excellent', 'Good', 'Average'],
                'values': [325, 587, 88]  # Scaled from dashboard
            }

    if 'genre' in query or 'homme/femme' in query or 'distribution par genre':
        if mode == 'b2c' and 'gender' in df.columns:
            counts = df['gender'].value_counts().dropna()
            return f"👥 **Distribution par genre :** {', '.join([f'{k} {round(v / len(df) * 100, 1)}%' for k,v in counts.items()])}.", {
                'chart_type': 'pie',
                'title': 'Distribution par Genre',
                'labels': counts.index.tolist(),
                'values': counts.values.tolist()
            }

    if 'statut relationnel' in query or 'relationship status':
        if mode == 'b2c' and 'relationship_status' in df.columns:
            counts = df['relationship_status'].value_counts().dropna()
            return f"💕 **Statuts relationnels :** {', '.join([f'{k} {round(v / len(df) * 100, 1)}%' for k,v in counts.items()])}.", {
                'chart_type': 'pie',
                'title': 'Statuts Relationnels',
                'labels': counts.index.tolist(),
                'values': counts.values.tolist()
            }

    if 'emploi' in query or 'taux d\'emploi' in query or 'statistics d\'emploi':
        if mode == 'b2c' and 'employment' in df.columns:
            employed = df['employment'].notna().sum()
            rate = round(employed / len(df) * 100, 1)
            counts = df['employment'].value_counts().dropna()
            return f"💼 **Taux d'emploi :** {rate}%. Clients employés : {employed}.\nTop lieux de travail : {', '.join([f'{k} ({v})' for k,v in counts.head(5).items()])}.", {
                'chart_type': 'pie',
                'title': 'Lieux de Travail',
                'labels': counts.index.tolist(),
                'values': counts.values.tolist()
            }

    if 'migration' in query or 'analyse de migration':
        if mode == 'b2c' and 'origin_city' in df.columns:
            with_origin = df['origin_city'].notna().sum()
            migration = len(df[df['origin_city'] != 'Paris'])  # Assume current is Paris
            rate = round(migration / len(df) * 100, 1)
            return f"✈️ **Analyse de migration :** Taux de migration : {rate}%. Clients avec données d'origine : {with_origin}.", {
                'chart_type': 'bar',
                'title': 'Analyse de Migration',
                'labels': ['Avec Migration', 'Sans Migration'],
                'values': [migration, len(df) - migration]
            }

    if 'complétude' in query or 'data quality' in query or 'qualité des données':
        if mode == 'b2b':
            rating_complete = round(df['rating'].notna().mean() * 100)
            score_complete = round(df['score'].notna().mean() * 100)
            reviews_complete = round(df['reviews'].notna().mean() * 100)
            website_complete = round(df['website'].notna().mean() * 100)
            return f"📊 **Qualité des données :** Avec Rating {rating_complete}%, Avec Score {score_complete}%, Avec Reviews {reviews_complete}%, Avec Website {website_complete}%.", {
                'chart_type': 'bar',
                'title': 'Qualité des Données',
                'labels': ['Rating', 'Score', 'Reviews', 'Website'],
                'values': [rating_complete, score_complete, reviews_complete, website_complete]
            }
        elif mode == 'b2c':
            work_complete = round(df['employment'].notna().mean() * 100)
            rel_complete = round(df['relationship_status'].notna().mean() * 100)
            origin_complete = round(df['origin_city'].notna().mean() * 100)
            phone_complete = round(df['phone'].notna().mean() * 100)
            return f"📊 **Complétude des données clients :** Avec Lieu de Travail {work_complete}%, Avec Statut Relationnel {rel_complete}%, Avec Ville d'Origine {origin_complete}%, Avec Téléphone {phone_complete}%.", {
                'chart_type': 'bar',
                'title': 'Complétude des Données Clients',
                'labels': ['Lieu de Travail', 'Statut Relationnel', 'Ville d\'Origine', 'Téléphone'],
                'values': [work_complete, rel_complete, origin_complete, phone_complete]
            }

    # Default
    return f"🔎 Je peux analyser les données {mode.upper()} sur : total, catégories/types, notes/ratings, villes, qualité, potentiel de croissance, genre, statut relationnel, emploi, migration, complétude des données. Posez une question spécifique !", {
        'chart_type': 'bar',
        'title': f'Analyse {mode.upper()} prête',
        'labels': ['Données'],
        'values': [len(df)]
    }

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'data_loaded': True,
        'b2b_count': len(dataframes['b2b']),
        'b2c_count': len(dataframes['b2c'])
    })

if __name__ == '__main__':
    print("🚀 Démarrage du serveur d'analyse B2B/B2C avec données intégrées...")
    app.run(port=5000, debug=True)