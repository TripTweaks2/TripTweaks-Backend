from flask import Flask, request, jsonify
import pandas as pd
import requests
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

def crear_app():

    app = Flask(__name__)

    # Cargar los datos con coordenadas
    data = pd.read_excel('Dataset_TripTweaks.xlsx')

    # Filtrar filas con coordenadas válidas
    data = data.dropna(subset=['LATITUD', 'LONGITUD'])

    # Definir las características y el objetivo
    features = ['LATITUD', 'LONGITUD', 'CATEGORIA']
    X = data[features]

    # Preprocesar datos
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['LATITUD', 'LONGITUD']),
            ('cat', OneHotEncoder(), ['CATEGORIA'])
        ])

    # Crear el pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    X_processed = pipeline.fit_transform(X)

    # Entrenar el modelo KNN
    model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    model.fit(X_processed)

    def recommend_places(latitude, longitude, category, model, pipeline, data, n_recommendations=5):
        # Crear un DataFrame con la entrada del usuario
        user_input = pd.DataFrame({
            'LATITUD': [latitude],
            'LONGITUD': [longitude],
            'CATEGORIA': [category]
        })
        
        # Preprocesar la entrada del usuario
        user_input_processed = pipeline.transform(user_input)
        
        # Obtener las recomendaciones
        distances, indices = model.kneighbors(user_input_processed, n_neighbors=n_recommendations)
        
        # Devolver las recomendaciones
        recommended_places = data.iloc[indices[0]]
        return recommended_places

    @app.route('/recommend', methods=['GET'])
    def recommend():
        latitude = float(request.args.get('LATITUD'))
        longitude = float(request.args.get('LONGITUD'))
        category = request.args.get('CATEGORIA')
        
        recommendations = recommend_places(latitude, longitude, category, model, pipeline, data)
        
        # Convertir a JSON
        recommendations_json = recommendations.to_dict(orient='records')
        
        return jsonify(recommendations_json)
    return app

if __name__ == '__main__':
    app =crear_app()
    app.run()
