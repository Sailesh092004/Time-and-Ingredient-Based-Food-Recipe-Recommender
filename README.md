# Vintage Flavours - Indian Recipe Recommender

A smart recipe recommendation system that suggests Indian recipes based on your preferences, available ingredients, and time constraints.

## Author
**Sailesh** **Sai Kiran Reddy** **Anisha Reddy**

## Features

- **Smart Recipe Recommendations**: Get personalized recipe suggestions based on multiple criteria
- **Ingredient-Based Matching**: Find recipes that match your available ingredients
- **Time-Aware**: Filter recipes based on preparation and cooking time constraints
- **Dietary Preferences**: Support for both vegetarian and non-vegetarian options
- **Regional Cuisine**: Filter recipes by North, South, East, West Indian cuisines, or explore all regions
- **Flavor Profiles**: Choose from different flavor profiles (Sweet, Spicy, Sour, Bitter)
- **Modern UI**: Clean and responsive user interface for easy interaction

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, Bootstrap
- **Machine Learning**: scikit-learn (TF-IDF Vectorization, Nearest Neighbors)
- **Data Processing**: pandas, numpy

## Project Structure

```
time_based_recipe_recommender/
├── app.py                      # Flask web application
├── recipe_generator.py         # Core recommendation logic
├── templates/
│   └── index.html             # Frontend interface
└── dataset/
    └── IndianFoodDatasetCSV.csv  # Recipe dataset
```

## How It Works

1. **Feature Engineering**:
   - Ingredients are vectorized using TF-IDF
   - Categorical features (diet, region, flavor) are encoded
   - Time features are scaled
   - Features are weighted based on importance:
     * Ingredients (6.0)
     * Diet (3.0)
     * Flavor (2.5)
     * Region (2.0)
     * Prep Time (1.5)
     * Cook Time (1.0)

2. **Recipe Matching**:
   - Uses cosine similarity to find nearest neighbor recipes
   - Considers all constraints while finding matches
   - Falls back gracefully when exact matches aren't found

## Setup and Running

1. **Clone the repository**:
   ```bash
   git clone [repository-url]
   cd faded_flavours
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python time_based_recipe_recommender/app.py
   ```

4. **Access the web interface**:
   - Open your browser and go to `http://localhost:5000`
   - Enter your preferences and get personalized recipe recommendations

## Usage

1. **Select Meal Type**: Choose from available Indian meal types
2. **Set Time Constraints**: Specify your preparation and cooking time limits
3. **Choose Diet**: Select vegetarian or non-vegetarian
4. **Pick Flavor Profile**: Choose your preferred flavor (Sweet, Spicy, Sour, Bitter)
5. **Select Region**: Choose a specific Indian region or "Any Region"
6. **Enter Ingredients**: List your available ingredients
7. **Get Recommendations**: Click "Generate Recipes" to get personalized suggestions

## Future Improvements

- Add user accounts and favorite recipes
- Include recipe ratings and reviews
- Add more sophisticated ingredient matching
- Implement recipe difficulty levels
- Add nutritional information
- Include cooking instructions and tips

## License

This project is licensed under the MIT License - see the LICENSE file for details.
