import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

# Configuration
np.random.seed(42)
random.seed(42)



# ==================== 1. GÉNÉRATION DU DATASET ====================
print("\n ÉTAPE 1: Génération du dataset")

# Services
services = pd.DataFrame({
    'service_id': range(1, 31),
    'titre': [
        'Génération de contenu SEO', 'Rédaction automatique d\'articles', 'Résumé de documents IA',
        'Correction grammaticale Pro', 'Traduction multilingue AI', 'Analyse de sentiment social',
        'Description produits e-commerce', 'Extraction mots-clés SEO', 'Génération d\'images IA',
        'Reconnaissance objet', 'Amélioration photo HD', 'Suppression arrière-plan',
        'Création visuels sociaux', 'Analyse vidéo intelligente', 'Prédiction ventes ML',
        'Analyse data clients', 'Détection anomalies', 'Recommandation e-commerce',
        'Segmentation clients', 'Dashboard prédictif', 'Chatbot support client',
        'Assistant RH virtuel', 'Automation emails', 'Gestion tickets support',
        'Planning posts sociaux', 'Génération scripts vidéo', 'Création avatars virtuels',
        'Montage vidéo auto', 'Optimisation CV IA', 'Génération devis auto'
    ],
    'categorie': [
        'NLP', 'NLP', 'NLP', 'NLP', 'NLP', 'Analyse', 'Commerce', 'SEO', 'Vision',
        'Vision', 'Vision', 'Vision', 'Design', 'Vision', 'Data', 'Data', 'Data',
        'Data', 'Data', 'Data', 'Chatbot', 'Chatbot', 'Automation', 'Support',
        'Social', 'Creation', 'Creation', 'Creation', 'Freelance', 'Business'
    ],
    'description': [
        'Génère automatiquement du contenu optimisé pour le SEO',
        'Rédige des articles de blog de qualité professionnelle',
        'Résume automatiquement des documents longs et complexes',
        'Corrige la grammaire et l\'orthographe en temps réel',
        'Traduit du contenu dans plus de 100 langues',
        'Analyse les sentiments sur les réseaux sociaux',
        'Génère des descriptions produits optimisées',
        'Extrait les mots-clés les plus pertinents',
        'Crée des images à partir de descriptions texte',
        'Détecte et reconnaît des objets sur images',
        'Améliore automatiquement la qualité des photos',
        'Supprime les arrière-plans en un clic',
        'Crée des visuels pour les réseaux sociaux',
        'Analyse le contenu vidéo automatiquement',
        'Prédit les ventes avec l\'apprentissage automatique',
        'Analyse en profondeur les données clients',
        'Détecte les anomalies dans les données',
        'Système de recommandation personnalisée',
        'Segmente automatiquement les clients',
        'Tableau de bord prédictif et analytics',
        'Chatbot intelligent pour le service client',
        'Assistant virtuel pour les RH',
        'Automatise l\'envoi d\'emails marketing',
        'Gère intelligemment les tickets support',
        'Planifie les posts sur les réseaux sociaux',
        'Génère des scripts vidéo automatiquement',
        'Crée des avatars virtuels personnalisés',
        'Montage vidéo automatisé',
        'Optimise les CV avec l\'IA',
        'Génère des devis automatiquement'
    ],
    'tags_associes': [
        'seo contenu blog', 'article blog redaction', 'document pdf resume',
        'grammaire orthographe', 'traduction langue', 'analyse sentiment',
        'ecommerce produit', 'seo mot cle', 'image design', 'objet detection',
        'photo amelioration', 'background image', 'social media design',
        'video analyse', 'vente prediction', 'client analyse', 'anomalie detection',
        'recommandation produit', 'segmentation client', 'dashboard predictif',
        'chatbot support', 'rh assistant', 'email automation', 'ticket support',
        'planification social', 'script video', 'avatar creation', 'montage video',
        'cv optimisation', 'devis estimation'
    ],
    'prix_mensuel': np.random.choice([19.99, 29.99, 49.99, 99.99, 199.99, 299.99], 30),
    'popularite': np.random.randint(50, 1000, 30),
    'note_moyenne': np.random.uniform(3.5, 5.0, 30).round(1)
})

# Demandes utilisateur
print("   - Génération des demandes...")
demandes = []
for i in range(500):
    templates = [
        f"Je cherche à automatiser la création de contenu pour les réseaux sociaux",
        f"Besoin d'un outil d'IA pour analyser les données clients",
        f"Comment améliorer mon SEO avec l'intelligence artificielle ?",
        f"Je veux un chatbot pour mon site e-commerce",
        f"Recherche solution IA pour prédire mes ventes",
        f"J'ai besoin d'un outil pour générer des images automatiquement"
    ]
    
    demandes.append({
        'request_id': i + 1,
        'description': random.choice(templates),
        'secteur': random.choice(['Marketing Digital', 'E-commerce', 'Services', 'Education', 'Sante', 'Finance']),
        'taille_utilisateur': random.choice(['Freelance', '1-10', '11-50', '51-200', '200+']),
        'urgence': random.choice(['Faible', 'Moyenne', 'Élevée']),
        'budget_estime': random.choice(['<100€', '100-500€', '500-2000€', '2000-5000€', '5000€+']),
        'date_creation': datetime.now() - timedelta(days=random.randint(0, 180))
    })

df_requests = pd.DataFrame(demandes)

# Interactions
print("   - Génération des interactions...")
interactions = []
action_types = ['view', 'click_detail', 'request_demo', 'request_submit']
for i in range(2000):
    interactions.append({
        'interaction_id': i + 1,
        'user_id': random.randint(1, 200),
        'service_id': random.randint(1, 30),
        'type_action': random.choice(action_types),
        'timestamp': datetime.now() - timedelta(days=random.randint(0, 180)),
        'temps_passe_seconde': random.randint(5, 600)
    })

df_interactions = pd.DataFrame(interactions)

# Sauvegarde
services.to_csv('mediaTower_services.csv', index=False)
df_requests.to_csv('mediaTower_requests.csv', index=False)
df_interactions.to_csv('mediaTower_interactions.csv', index=False)

print(f"\n Dataset créé: {len(services)} services, {len(df_requests)} demandes, {len(df_interactions)} interactions")

# ==================== 2. DATA PREPROCESSING ====================
print("\n ÉTAPE 2: Prétraitement des données")

class DataPreprocessor:
    def __init__(self):
        self.minmax_scaler = MinMaxScaler()
        self.label_encoders = {}
        
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess_services(self, df):
        df = df.copy()
        df['clean_titre'] = df['titre'].apply(self.clean_text)
        df['clean_description'] = df['description'].apply(self.clean_text)
        df['clean_tags'] = df['tags_associes'].apply(self.clean_text)
        df['text_features'] = df['clean_titre'] + " " + df['clean_description'] + " " + df['clean_tags']
        
        self.label_encoders['categorie'] = LabelEncoder()
        df['categorie_encoded'] = self.label_encoders['categorie'].fit_transform(df['categorie'])
        df['prix_normalized'] = self.minmax_scaler.fit_transform(df[['prix_mensuel']])
        df['popularite_normalized'] = self.minmax_scaler.fit_transform(df[['popularite']])
        return df

preprocessor = DataPreprocessor()
services_processed = preprocessor.preprocess_services(services)
print("   ✓ Services prétraités")

# ==================== 3. FEATURE ENGINEERING (CORRIGÉ) ====================
print("\n ÉTAPE 3: Feature engineering")

class FeatureEngineer:
    def __init__(self, n_components=20):
        # Liste de stop words français personnalisée (simple)
        self.french_stop_words = [
            'le', 'la', 'les', 'un', 'une', 'des', 'pour', 'par', 'avec', 'sans',
            'est', 'sont', 'et', 'ou', 'mais', 'donc', 'car', 'de', 'du', 'des',
            'ce', 'cette', 'ces', 'il', 'elle', 'ils', 'elles', 'on', 'nous',
            'vous', 'qui', 'que', 'quoi', 'dont', 'ou', 'lui', 'leur', 'leurs'
        ]
        self.tfidf = TfidfVectorizer(max_features=100, stop_words=self.french_stop_words)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
    def create_service_embeddings(self, df):
        tfidf_matrix = self.tfidf.fit_transform(df['text_features'])
        embeddings = self.svd.fit_transform(tfidf_matrix)
        return embeddings
    
    def create_user_profiles(self, interactions_df, embeddings):
        user_profiles = {}
        weights = {'view': 1, 'click_detail': 2, 'request_demo': 3, 'request_submit': 4}
        
        for user_id in interactions_df['user_id'].unique():
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            weighted = []
            
            for _, interaction in user_interactions.iterrows():
                service_id = interaction['service_id'] - 1
                if service_id < len(embeddings):
                    weight = weights.get(interaction['type_action'], 1)
                    weighted.append(embeddings[service_id] * weight)
            
            if weighted:
                user_profiles[user_id] = np.mean(weighted, axis=0)
            else:
                user_profiles[user_id] = np.zeros(embeddings.shape[1])
        
        return user_profiles

feature_engineer = FeatureEngineer()
service_embeddings = feature_engineer.create_service_embeddings(services_processed)
user_profiles = feature_engineer.create_user_profiles(df_interactions, service_embeddings)
print(f"   ✓ Embeddings créés: {service_embeddings.shape}")
print(f"   ✓ Profils utilisateur: {len(user_profiles)}")

# ==================== 4. SYSTÈME DE RECOMMANDATION ====================
print("\n ÉTAPE 4: Création du système de recommandation")

class ContentBasedRecommender:
    def __init__(self):
        # Liste de stop words français
        french_stop_words = [
            'le', 'la', 'les', 'un', 'une', 'des', 'pour', 'par', 'avec', 'sans',
            'est', 'sont', 'et', 'ou', 'mais', 'donc', 'car', 'de', 'du', 'des',
            'ce', 'cette', 'ces', 'il', 'elle', 'ils', 'elles', 'on', 'nous',
            'vous', 'qui', 'que', 'quoi', 'dont', 'ou', 'lui', 'leur', 'leurs'
        ]
        self.tfidf = TfidfVectorizer(stop_words=french_stop_words)
        self.similarity_matrix = None
        self.services_df = None
        
    def fit(self, services_df):
        self.services_df = services_df
        text_features = services_df['clean_titre'] + " " + services_df['clean_description']
        tfidf_matrix = self.tfidf.fit_transform(text_features)
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        
    def recommend_by_description(self, description, top_n=5):
        desc_vector = self.tfidf.transform([description])
        similarities = cosine_similarity(desc_vector, self.tfidf.transform(
            self.services_df['clean_titre'] + " " + self.services_df['clean_description']
        )).flatten()
        
        top_indices = similarities.argsort()[-top_n:][::-1]
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'service_id': self.services_df.iloc[idx]['service_id'],
                'titre': self.services_df.iloc[idx]['titre'],
                'categorie': self.services_df.iloc[idx]['categorie'],
                'prix': self.services_df.iloc[idx]['prix_mensuel'],
                'score': similarities[idx],
                'description': self.services_df.iloc[idx]['description']
            })
        return recommendations

class HybridRecommender:
    def __init__(self):
        self.content_recommender = ContentBasedRecommender()
        self.services_df = None
        
    def fit(self, services_df):
        self.services_df = services_df
        self.content_recommender.fit(services_df)
        
    def recommend(self, user_id=None, description=None, top_n=5):
        if description:
            return self.content_recommender.recommend_by_description(description, top_n)
        else:
            popular = self.services_df.nlargest(top_n, 'popularite')
            return popular[['service_id', 'titre', 'categorie', 'prix_mensuel', 'description']].to_dict('records')
    
    def recommend_by_need_with_analysis(self, description, top_n=5):
        recs = self.content_recommender.recommend_by_description(description, top_n)
        analysis = self._analyze_need(description)
        return recs, analysis
    
    def _analyze_need(self, description):
        desc_lower = description.lower()
        categories = []
        if any(word in desc_lower for word in ['contenu', 'créer', 'générer', 'rédiger', 'contenu']):
            categories.append('creation')
        if any(word in desc_lower for word in ['analyser', 'prédire', 'évaluer', 'analyse']):
            categories.append('analyse')
        if any(word in desc_lower for word in ['chatbot', 'conversation', 'assistant', 'support']):
            categories.append('chatbot')
        if any(word in desc_lower for word in ['image', 'photo', 'visuel', 'design', 'visuel']):
            categories.append('vision')
        if any(word in desc_lower for word in ['vente', 'client', 'data', 'donnée', 'client']):
            categories.append('data')
        if any(word in desc_lower for word in ['seo', 'référencement', 'trafic']):
            categories.append('seo')
        
        return {
            'categories': categories if categories else ['general'],
            'has_urgence': 'urgent' in desc_lower or 'rapide' in desc_lower,
            'complexite': 'élevée' if 'complexe' in desc_lower else 'moyenne'
        }

# ==================== 5. ANALYSE NLP ====================
print("\n ÉTAPE 5: Analyse NLP des demandes")

class NLPProcessor:
    def __init__(self):
        self.keyword_categories = {
            'creation': ['créer', 'générer', 'produire', 'rédiger', 'contenu', 'article', 'contenu'],
            'analyse': ['analyser', 'prédire', 'évaluer', 'mesurer', 'data', 'donnée', 'analyse'],
            'chatbot': ['chatbot', 'conversation', 'assistant', 'support', 'client', 'service client'],
            'vision': ['image', 'photo', 'visuel', 'design', 'vidéo', 'visuel'],
            'seo': ['seo', 'référencement', 'trafic', 'mots clés', 'référencement']
        }
    
    def analyze(self, text):
        text_lower = text.lower()
        detected_categories = []
        keywords_found = []
        
        for category, keywords in self.keyword_categories.items():
            found = [kw for kw in keywords if kw in text_lower]
            if found:
                detected_categories.append(category)
                keywords_found.extend(found)
        
        return {
            'categories': detected_categories if detected_categories else ['general'],
            'keywords': list(set(keywords_found))[:5],
            'confidence': len(detected_categories) / len(self.keyword_categories) if detected_categories else 0.3
        }

nlp_processor = NLPProcessor()

# ==================== 6. ONLINE LEARNING ====================
print("\n ÉTAPE 6: Configuration apprentissage en ligne")

class OnlineRecommender:
    def __init__(self, decay_factor=0.95):
        self.decay_factor = decay_factor
        self.user_scores = defaultdict(lambda: defaultdict(float))
        self.service_popularity = defaultdict(float)
        
    def update(self, user_id, service_id, action):
        weights = {'view': 1, 'click_detail': 3, 'request_demo': 5, 'request_submit': 10}
        weight = weights.get(action, 1)
        self.user_scores[user_id][service_id] += weight
        self.service_popularity[service_id] += weight
        self._apply_decay()
    
    def _apply_decay(self):
        for user in self.user_scores:
            for service in self.user_scores[user]:
                self.user_scores[user][service] *= self.decay_factor
        for service in self.service_popularity:
            self.service_popularity[service] *= self.decay_factor
    
    def get_recommendations(self, user_id, top_n=5):
        if user_id in self.user_scores:
            scores = self.user_scores[user_id]
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        else:
            return sorted(self.service_popularity.items(), key=lambda x: x[1], reverse=True)[:top_n]

online_recommender = OnlineRecommender()

# Simulation d'apprentissage
print("   - Simulation d'apprentissage en ligne...")
for _, interaction in df_interactions.iterrows():
    online_recommender.update(interaction['user_id'], interaction['service_id'], interaction['type_action'])
print("   ✓ Apprentissage en ligne initialisé")

# ==================== 7. INITIALISATION SYSTÈME HYBRIDE ====================
print("\n ÉTAPE 7: Initialisation du système hybride...")

hybrid_recommender = HybridRecommender()
hybrid_recommender.fit(services_processed)
print("   ✓ Système hybride prêt")

# ==================== 8. DÉMONSTRATION ====================
print("\n" + "="*70)
print(" DÉMONSTRATION DU SYSTÈME")
print("="*70)

# Test 1: Recommandation par description
print("\n TEST 1: Recommandation par description de besoin")
description = "Je cherche un outil pour créer du contenu automatique pour mes réseaux sociaux"
print(f"   Besoin: '{description}'")

nlp_result = nlp_processor.analyze(description)
print(f"   Analyse NLP: Catégories={nlp_result['categories']}, Mots-clés={nlp_result['keywords']}")

recommendations, analysis = hybrid_recommender.recommend_by_need_with_analysis(description, top_n=5)
print("\n    Top 5 recommandations:")
for i, rec in enumerate(recommendations, 1):
    print(f"   {i}. {rec['titre']} ({rec['categorie']}) - {rec['prix']}€ - Score: {rec['score']:.3f}")
    print(f"      {rec['description'][:60]}...")

# Test 2: Services populaires
print("\n TEST 2: Top services populaires")
popular = hybrid_recommender.recommend(top_n=5)
for i, rec in enumerate(popular, 1):
    print(f"   {i}. {rec['titre']} ({rec['categorie']}) - {rec['prix_mensuel']}€")

# Test 3: Recommandation online
print("\n TEST 3: Recommandation online pour utilisateur 1")
online_recs = online_recommender.get_recommendations(1, top_n=5)
for i, (service_id, score) in enumerate(online_recs, 1):
    service = services[services['service_id'] == service_id].iloc[0]
    print(f"   {i}. {service['titre']} - Score: {score:.1f}")

# ==================== 9. STATISTIQUES ====================
print("\n" + "="*70)
print(" STATISTIQUES DU SYSTÈME")
print("="*70)

print(f"\n Distribution des services par catégorie:")
for cat, count in services['categorie'].value_counts().items():
    bar = "" * int(count / 2)
    print(f"   {cat:12} : {count:2} services {bar}")

print(f"\n Prix moyens par catégorie:")
for cat in services['categorie'].unique():
    avg_price = services[services['categorie'] == cat]['prix_mensuel'].mean()
    print(f"   {cat:12} : {avg_price:.2f}€/mois")

print(f"\n Top 5 services les mieux notés:")
top_rated = services.nlargest(5, 'note_moyenne')[['titre', 'note_moyenne', 'popularite']]
for _, row in top_rated.iterrows():
    print(f"   {row['titre'][:30]:30s}  {row['note_moyenne']}/5 ( {row['popularite']})")

# ==================== 10. ANALYSE DES DATASETS ====================
print("\n" + "="*70)
print(" ANALYSE DES DATASETS")
print("="*70)

datasets = {
    'Services': services,
    'Demandes': df_requests,
    'Interactions': df_interactions
}

total_lignes = 0
for name, df in datasets.items():
    print(f"\n {name}: {len(df):,} lignes, {len(df.columns)} colonnes")
    print(f"   Colonnes: {', '.join(df.columns[:5])}...")
    total_lignes += len(df)

print(f"\n TOTAL GÉNÉRAL: {total_lignes:,} lignes sur {len(datasets)} fichiers")

# ==================== 11. MODE INTERACTIF ====================
print("\n" + "="*70)
print(" MODE INTERACTIF")
print("="*70)

choice = input("\nVoulez-vous tester le système en mode interactif? (o/n): ")

if choice.lower() == 'o':
    print("\n Mode interactif - Entrez vos besoins en IA")
    print("   Exemples: 'Je cherche un chatbot pour mon site'")
    print("            'Besoin d'analyser mes données clients'")
    print("            'Générer des images automatiquement'")
    print("   (tapez 'quit' pour quitter)")
    print("-"*60)
    
    while True:
        user_need = input("\n Décrivez votre besoin: ")
        if user_need.lower() == 'quit':
            print("\n Merci d'avoir utilisé MediaTower!")
            break
        
        if user_need.strip():
            print("\n Analyse en cours...")
            nlp_analysis = nlp_processor.analyze(user_need)
            print(f"    Catégories détectées: {', '.join(nlp_analysis['categories'])}")
            print(f"    Mots-clés: {', '.join(nlp_analysis['keywords'])}")
            
            recs, _ = hybrid_recommender.recommend_by_need_with_analysis(user_need, top_n=5)
            print("\n Recommandations personnalisées:")
            print("   " + "-"*50)
            for i, rec in enumerate(recs, 1):
                print(f"   {i}. {rec['titre']}")
                print(f"       Catégorie: {rec['categorie']} |  Prix: {rec['prix']}€/mois")
                print(f"       {rec['description'][:80]}...")
                print(f"       Score de pertinence: {rec['score']:.2%}")
                print()

# ==================== 12. CONCLUSION ====================
print("\n" + "="*70)
print(" SYSTÈME DE RECOMMANDATION COMPLET - PRÊT À L'EMPLOI")
print("="*70)
print("\n Fonctionnalités implémentées:")
print("   ✓ Génération automatique du dataset (30 services, 500 demandes, 2000 interactions)")
print("   ✓ Prétraitement des données (nettoyage, normalisation)")
print("   ✓ Feature engineering (TF-IDF avec stop words français, SVD)")
print("   ✓ Système de recommandation content-based")
print("   ✓ Analyse NLP des besoins utilisateur")
print("   ✓ Apprentissage en ligne (online learning)")
print("   ✓ Mode interactif pour tester")
print("   ✓ Métriques et statistiques détaillées")
print("\n Fichiers CSV générés:")
print("   - mediaTower_services.csv")
print("   - mediaTower_requests.csv")
print("   - mediaTower_interactions.csv")
print("\n Pour utiliser le système:")
print("   1. Lancez ce script pour générer les données et le modèle")
print("   2. Utilisez le mode interactif pour tester des recommandations")
print("   3. Les recommandations sont basées sur l'analyse NLP et la similarité")
print("="*70)