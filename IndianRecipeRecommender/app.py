from flask import Flask, render_template, request, jsonify
from recipe_generator import RecipeGenerator
import os

app = Flask(__name__)
generator = RecipeGenerator('dataset/IndianFoodDatasetCSV.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_meal_types')
def get_meal_types():
    return jsonify(generator.get_available_meal_types())

@app.route('/generate_recipes', methods=['POST'])
def generate_recipes():
    constraints = request.json
    
    # Handle "any" region selection
    if 'region' in constraints and constraints['region'].lower() == 'any':
        constraints['region'] = None
    
    recommendations = generator.generate_recipe(constraints)
    return jsonify(recommendations)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True) 