import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
import re

class RecipeGenerator:
    def __init__(self, dataset_path):
        # Get the absolute path to the dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, dataset_path)
        
        # Load the dataset
        self.df = pd.read_csv(dataset_path)
        
        # Rename columns to match our expected format
        self.df = self.df.rename(columns={
            'TranslatedRecipeName': 'name',
            'TranslatedIngredients': 'ingredients',
            'PrepTimeInMins': 'prep_time',
            'CookTimeInMins': 'cook_time',
            'TotalTimeInMins': 'total_time',
            'Cuisine': 'region',
            'Course': 'course',
            'Diet': 'diet',
            'URL': 'recipe_link'
        })
        
        # Initialize ingredient mappings
        self._initialize_ingredient_mappings()
        
        # Initialize encoders and vectorizers
        self._initialize_encoders()
        
        # Preprocess the data
        self._preprocess_data()
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_ingredient_mappings(self):
        """Initialize mappings for ingredient variations and categories"""
        # Common Indian ingredient variations and synonyms
        self.ingredient_variations = {
            # Dairy and Proteins
            'paneer': ['panner', 'panir', 'cottage cheese', 'पनीर', 'cottage', 'cheese'],
            'ghee': ['clarified butter', 'घी', 'ghhee', 'ghee butter'],
            'dahi': ['yogurt', 'curd', 'yoghurt', 'दही', 'yogurt curd'],
            'milk': ['doodh', 'दूध', 'milk powder', 'powdered milk'],
            
            # Lentils and Pulses
            'dal': ['daal', 'dhal', 'dahl', 'lentils', 'पल्स', 'lentil'],
            'moong dal': ['mung dal', 'green gram', 'मूंग दाल', 'moong'],
            'toor dal': ['arhar dal', 'pigeon pea', 'तूर दाल', 'tuvar dal'],
            'chana dal': ['bengal gram', 'चना दाल', 'chana', 'gram dal'],
            
            # Spices
            'haldi': ['turmeric', 'turmeric powder', 'हल्दी', 'haldi powder'],
            'jeera': ['cumin', 'cumin seeds', 'जीरा', 'zeera'],
            'dhania': ['coriander', 'coriander powder', 'धनिया', 'coriander seeds'],
            'mirch': ['chilli', 'chili', 'chile', 'red chilli', 'मिर्च', 'chilli powder'],
            'garam masala': ['indian spice mix', 'गरम मसाला', 'spice mix'],
            'elaichi': ['cardamom', 'cardamom pods', 'इलायची', 'cardamom powder'],
            
            # Vegetables
            'aloo': ['potato', 'potatoes', 'आलू', 'potato cubes'],
            'pyaaz': ['onion', 'onions', 'प्याज', 'onion slices'],
            'tamatar': ['tomato', 'tomatoes', 'टमाटर', 'tomato puree'],
            'gobhi': ['cauliflower', 'गोभी', 'cauliflower florets'],
            'gajar': ['carrot', 'carrots', 'गाजर', 'carrot slices'],
            'mushroom': ['mashroom', 'mushrooms', 'मशरूम', 'button mushroom'],
            
            # Grains and Flours
            'atta': ['wheat flour', 'whole wheat flour', 'आटा', 'flour'],
            'maida': ['all purpose flour', 'refined flour', 'मैदा', 'plain flour'],
            'chawal': ['rice', 'चावल', 'basmati rice', 'rice grains'],
            'besan': ['gram flour', 'chickpea flour', 'बेसन', 'gram flour powder'],
            
            # Aromatics
            'lahsun': ['garlic', 'लहसुन', 'garlic cloves', 'garlic paste'],
            'adrak': ['ginger', 'अदरक', 'ginger paste', 'ginger root'],
            'pudina': ['mint', 'mint leaves', 'पुदीना', 'mint powder'],
            'dhaniya patta': ['coriander leaves', 'cilantro', 'धनिया पत्ता', 'coriander'],
            
            # Others
            'pani': ['water', 'पानी', 'hot water', 'cold water'],
            'namak': ['salt', 'नमक', 'rock salt', 'sea salt'],
            'cheeni': ['sugar', 'चीनी', 'powdered sugar', 'brown sugar'],
            'tel': ['oil', 'तेल', 'vegetable oil', 'cooking oil']
        }
        
        # Ingredient categories for better matching
        self.ingredient_categories = {
            'primary': [
                'chicken', 'mutton', 'fish', 'paneer', 'dal', 'rice',
                'potato', 'cauliflower', 'mushroom', 'egg'
            ],
            'spices': [
                'turmeric', 'cumin', 'coriander', 'garam masala', 'cardamom',
                'cinnamon', 'clove', 'pepper', 'chilli'
            ],
            'aromatics': [
                'onion', 'garlic', 'ginger', 'mint', 'coriander leaves'
            ],
            'dairy': [
                'ghee', 'butter', 'cream', 'yogurt', 'milk', 'paneer'
            ]
        }
        
        # Create a reverse lookup for faster matching
        self.ingredient_reverse_lookup = {}
        for standard, variations in self.ingredient_variations.items():
            for variation in variations:
                self.ingredient_reverse_lookup[variation] = standard

    def _normalize_ingredients(self, ingredients_text):
        """Normalize ingredient text by handling variations and synonyms"""
        ingredients_text = ingredients_text.lower()
        normalized_text = ingredients_text
        
        # First, try to match complete words
        words = re.findall(r'\b\w+\b', ingredients_text)
        for word in words:
            # Handle common misspellings
            if word == 'potatos':
                word = 'potato'
            elif word == 'onions':
                word = 'onion'
            elif word == 'mashroom':
                word = 'mushroom'
            elif word == 'panner':
                word = 'paneer'
            
            if word in self.ingredient_reverse_lookup:
                normalized_text = re.sub(r'\b' + re.escape(word) + r'\b', 
                                      self.ingredient_reverse_lookup[word], 
                                      normalized_text)
        
        # Then try to match partial words
        for standard, variations in self.ingredient_variations.items():
            pattern = '|'.join(map(re.escape, variations))
            normalized_text = re.sub(pattern, standard, normalized_text)
        
        return normalized_text

    def _calculate_ingredient_match_score(self, available_ingredients, recipe_ingredients):
        """Calculate a sophisticated ingredient match score"""
        # Normalize both ingredient lists
        available = self._normalize_ingredients(','.join(available_ingredients))
        recipe = self._normalize_ingredients(recipe_ingredients)
        
        # Convert to sets for matching
        available_set = set(available.split(','))
        recipe_set = set(recipe.split(','))
        
        # Calculate basic match score
        matched_ingredients = available_set.intersection(recipe_set)
        basic_score = len(matched_ingredients) / len(available_set) if available_set else 0
        
        # Calculate primary ingredient match bonus
        primary_matches = sum(1 for ing in matched_ingredients 
                            if any(p in ing for p in self.ingredient_categories['primary']))
        primary_bonus = primary_matches * 0.3  # 30% bonus for each primary ingredient match
        
        # Calculate spice match score
        spice_matches = sum(1 for ing in matched_ingredients 
                          if any(s in ing for s in self.ingredient_categories['spices']))
        spice_score = (spice_matches / len(self.ingredient_categories['spices'])) * 0.2
        
        # Add bonus for partial matches
        partial_matches = 0
        for avail_ing in available_set:
            for recipe_ing in recipe_set:
                if avail_ing in recipe_ing or recipe_ing in avail_ing:
                    partial_matches += 1
        partial_bonus = min(0.2, partial_matches * 0.1)  # Max 20% bonus for partial matches
        
        # Combine scores
        total_score = min(1.0, basic_score + primary_bonus + spice_score + partial_bonus)
        
        return total_score

    def _initialize_encoders(self):
        # Initialize encoders for categorical features
        self.diet_encoder = LabelEncoder()
        self.flavor_encoder = LabelEncoder()
        self.course_encoder = LabelEncoder()
        self.region_encoder = LabelEncoder()
        
        # Pre-fit encoders with known values
        self.diet_encoder.fit(['vegetarian', 'non-vegetarian'])
        self.flavor_encoder.fit(['sweet', 'spicy', 'sour', 'bitter'])
        self.region_encoder.fit(['North', 'South', 'East', 'West', 'Other'])
        
        # Initialize TF-IDF vectorizer for ingredients
        self.ingredient_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            token_pattern=r'\b\w+\b'
        )
    
    def _initialize_region_mappings(self):
        """Initialize mappings for region classification"""
        self.region_mappings = {
            'North': {
                'states': [
                    'punjab', 'haryana', 'delhi', 'uttar pradesh', 'uttarakhand',
                    'himachal pradesh', 'jammu', 'kashmir', 'rajasthan'
                ],
                'cuisines': [
                    'punjabi', 'north indian', 'kashmiri', 'rajasthani', 'awadhi',
                    'mughlai', 'up', 'delhi'
                ],
                'dishes': [
                    'butter chicken', 'dal makhani', 'chole', 'rajma', 'paratha',
                    'samosa', 'tandoori', 'biryani', 'kebab'
                ]
            },
            'South': {
                'states': [
                    'kerala', 'tamil nadu', 'karnataka', 'andhra pradesh',
                    'telangana', 'pondicherry'
                ],
                'cuisines': [
                    'south indian', 'kerala', 'tamil', 'andhra', 'chettinad',
                    'udupi', 'mangalorean', 'hyderabadi'
                ],
                'dishes': [
                    'dosa', 'idli', 'vada', 'sambar', 'rasam', 'pongal',
                    'appam', 'aviyal', 'biryani'
                ]
            },
            'East': {
                'states': [
                    'west bengal', 'odisha', 'bihar', 'jharkhand',
                    'assam', 'manipur', 'tripura', 'meghalaya'
                ],
                'cuisines': [
                    'bengali', 'odia', 'bihari', 'assamese', 'manipuri',
                    'northeast', 'kolkata'
                ],
                'dishes': [
                    'machher jhol', 'mishti doi', 'rasgulla', 'pitha',
                    'thukpa', 'momos', 'litti chokha'
                ]
            },
            'West': {
                'states': [
                    'gujarat', 'maharashtra', 'goa'
                ],
                'cuisines': [
                    'gujarati', 'maharashtrian', 'goan', 'parsi',
                    'malvani', 'mumbai'
                ],
                'dishes': [
                    'dhokla', 'thepla', 'vada pav', 'misal pav',
                    'vindaloo', 'xacuti', 'dhansak'
                ]
            }
        }
        
        # Create a combined lookup dictionary for easier matching
        self.region_lookup = {}
        for region, data in self.region_mappings.items():
            for category in data.values():
                for term in category:
                    self.region_lookup[term] = region

    def _classify_region(self, recipe_text):
        """Classify a recipe's region based on its name, ingredients, and other attributes"""
        recipe_text = recipe_text.lower()
        
        # Check for direct region matches
        for term, region in self.region_lookup.items():
            if term in recipe_text:
                return region
        
        # Check for ingredient-based patterns
        ingredient_patterns = {
            'North': ['ghee', 'cream', 'paneer', 'butter', 'tandoori'],
            'South': ['coconut', 'curry leaves', 'tamarind', 'idli', 'dosa'],
            'East': ['mustard oil', 'fish', 'posto', 'bamboo shoot'],
            'West': ['kokum', 'jaggery', 'coconut', 'peanuts']
        }
        
        region_scores = {region: 0 for region in self.region_mappings.keys()}
        
        for region, ingredients in ingredient_patterns.items():
            for ingredient in ingredients:
                if ingredient in recipe_text:
                    region_scores[region] += 1
        
        # Get the region with the highest score
        max_score = max(region_scores.values())
        if max_score > 0:
            return max(region_scores.items(), key=lambda x: x[1])[0]
        
        return 'Other'

    def _preprocess_data(self):
        """Preprocess the dataset"""
        # Initialize region mappings
        self._initialize_region_mappings()
        
        # Convert time columns to minutes
        for time_col in ['prep_time', 'cook_time']:
            self.df[time_col] = self.df[time_col].fillna(0).astype(float)
        
        # Clean and standardize ingredients
        self.df['ingredients'] = self.df['ingredients'].fillna('')
        self.df['ingredients'] = self.df['ingredients'].astype(str)
        self.df['ingredients'] = self.df['ingredients'].apply(self._normalize_ingredients)
        
        # Classify regions based on recipe text
        self.df['region'] = self.df.apply(
            lambda row: self._classify_region(
                f"{row['name']} {row['ingredients']} {row['region']}"
            ),
            axis=1
        )
        
        # Set flavor profile based on ingredients and course type
        dessert_ingredients = ['sugar', 'jaggery', 'honey', 'chocolate', 'sweet', 'candy', 'condensed milk']
        self.df['flavor_profile'] = 'spicy'  # default
        self.df.loc[self.df['ingredients'].str.contains('|'.join(dessert_ingredients), na=False), 'flavor_profile'] = 'sweet'
        self.df.loc[self.df['course'].str.lower().str.contains('dessert|sweet|mithai|mishti', na=False), 'flavor_profile'] = 'sweet'
        
        # Standardize diet categories
        self.df['diet'] = self.df['diet'].str.lower()
        self.df.loc[self.df['diet'].str.contains('non-vegetarian|non vegetarian|egg|chicken|fish|meat', na=False), 'diet'] = 'non-vegetarian'
        self.df.loc[~self.df['diet'].str.contains('non-vegetarian|non vegetarian|egg|chicken|fish|meat', na=False), 'diet'] = 'vegetarian'
        
        # Encode categorical features
        self.df['diet_encoded'] = self.diet_encoder.transform(self.df['diet'])
        self.df['flavor_encoded'] = self.flavor_encoder.transform(self.df['flavor_profile'])
        self.df['course_encoded'] = self.course_encoder.fit_transform(self.df['course'])
        self.df['region_encoded'] = self.region_encoder.transform(self.df['region'])
        
        # Vectorize ingredients
        self.ingredient_vectors = self.ingredient_vectorizer.fit_transform(self.df['ingredients'])
        
        # Create feature matrix
        prep_time = self.df[['prep_time']].values
        cook_time = self.df[['cook_time']].values
        diet_features = np.eye(2)[self.df['diet_encoded']]
        flavor_features = np.eye(len(self.flavor_encoder.classes_))[self.df['flavor_encoded']]
        region_features = np.eye(len(self.region_encoder.classes_))[self.df['region_encoded']]
        
        # Scale time features
        self.time_scaler = StandardScaler()
        scaled_prep_time = self.time_scaler.fit_transform(prep_time)
        scaled_cook_time = self.time_scaler.fit_transform(cook_time)
        
        # Create ingredient match scores matrix
        ingredient_scores = np.zeros((len(self.df), 1))
        
        # Combine all features with weights
        self.features = np.hstack([
            self.ingredient_vectors.toarray() * 6.0,  # Highest weight for ingredients
            diet_features * 3.0,     # Diet is important
            flavor_features * 2.5,    # Flavor is next
            region_features * 2.0,    # Region is less important
            scaled_prep_time * 1.5,   # Prep time is less strict
            scaled_cook_time * 1.0    # Cook time is least strict
        ])
    
    def _initialize_model(self):
        self.model = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='cosine')
        self.model.fit(self.features)
    
    def get_available_meal_types(self):
        return sorted(self.df['course'].unique().tolist())
    
    def _filter_by_constraints(self, constraints):
        """Filter recipes based on user constraints"""
        filtered_df = self.df.copy()
        
        # Debug print
        print(f"\nInitial recipes: {len(filtered_df)}")
        
        # Filter by course type
        if 'course' in constraints and constraints['course']:
            filtered_df = filtered_df[filtered_df['course'] == constraints['course']]
            print(f"After course filter: {len(filtered_df)}")
        
        # Filter by diet
        if 'diet' in constraints and constraints['diet']:
            filtered_df = filtered_df[filtered_df['diet'] == constraints['diet']]
            print(f"After diet filter: {len(filtered_df)}")
        
        # Filter by region
        if 'region' in constraints and constraints['region']:
            filtered_df = filtered_df[filtered_df['region'] == constraints['region']]
            print(f"After region filter: {len(filtered_df)}")
        
        # Filter by time constraints
        if 'prep_time' in constraints:
            filtered_df = filtered_df[filtered_df['prep_time'] <= constraints['prep_time'] * 1.2]  # Allow 20% flexibility
        if 'cook_time' in constraints:
            filtered_df = filtered_df[filtered_df['cook_time'] <= constraints['cook_time'] * 1.2]
        print(f"After time filter: {len(filtered_df)}")
        
        # Filter by ingredients
        if 'ingredients' in constraints and constraints['ingredients']:
            # Normalize user ingredients
            user_ingredients = [self._normalize_ingredients(ing) for ing in constraints['ingredients']]
            
            # Calculate ingredient match scores
            filtered_df['ingredient_match'] = filtered_df['ingredients'].apply(
                lambda x: self._calculate_ingredient_match_score(user_ingredients, x)
            )
            
            # Filter recipes with at least one matching ingredient
            filtered_df = filtered_df[filtered_df['ingredient_match'] > 0]
            print(f"After ingredient filter: {len(filtered_df)}")
            
            # Sort by ingredient match score
            filtered_df = filtered_df.sort_values('ingredient_match', ascending=False)
        
        # If no recipes match all constraints, try relaxing some constraints
        if len(filtered_df) == 0:
            print("\nNo recipes match all constraints. Relaxing some constraints...")
            
            # Start with original dataset
            filtered_df = self.df.copy()
            
            # Keep diet and region constraints strict
            if 'diet' in constraints and constraints['diet']:
                filtered_df = filtered_df[filtered_df['diet'] == constraints['diet']]
            
            if 'region' in constraints and constraints['region']:
                filtered_df = filtered_df[filtered_df['region'] == constraints['region']]
            
            # Relax time constraints more
            if 'prep_time' in constraints:
                filtered_df = filtered_df[filtered_df['prep_time'] <= constraints['prep_time'] * 1.5]
            if 'cook_time' in constraints:
                filtered_df = filtered_df[filtered_df['cook_time'] <= constraints['cook_time'] * 1.5]
            
            # Calculate ingredient matches
            if 'ingredients' in constraints and constraints['ingredients']:
                user_ingredients = [self._normalize_ingredients(ing) for ing in constraints['ingredients']]
                filtered_df['ingredient_match'] = filtered_df['ingredients'].apply(
                    lambda x: self._calculate_ingredient_match_score(user_ingredients, x)
                )
                # Lower the threshold for ingredient matches when relaxing constraints
                filtered_df = filtered_df[filtered_df['ingredient_match'] > 0.1]  # Changed from 0 to 0.1
                filtered_df = filtered_df.sort_values('ingredient_match', ascending=False)
            
            print(f"After relaxing constraints: {len(filtered_df)} recipes found")
        
        return filtered_df

    def generate_recipe(self, constraints):
        """Generate recipes based on user constraints"""
        # First filter recipes by constraints
        filtered_df = self._filter_by_constraints(constraints)
        
        if len(filtered_df) == 0:
            return []
        
        # Create input feature vector
        input_features = np.zeros(self.features.shape[1])
        current_pos = 0
        
        # Add ingredient features
        ingredients_text = ', '.join(constraints.get('ingredients', []))
        ingredient_vector = self.ingredient_vectorizer.transform([ingredients_text]).toarray()[0]
        ingredient_dim = len(ingredient_vector)
        input_features[current_pos:current_pos + ingredient_dim] = ingredient_vector * 6.0
        current_pos += ingredient_dim
        
        # Add diet feature
        diet_dim = 2
        if 'diet' in constraints and constraints['diet']:
            diet_idx = self.diet_encoder.transform([constraints['diet']])[0]
            diet_vector = np.eye(2)[diet_idx]
            input_features[current_pos:current_pos + diet_dim] = diet_vector * 3.0
        current_pos += diet_dim
        
        # Add flavor feature
        flavor_dim = len(self.flavor_encoder.classes_)
        if 'flavor_profile' in constraints and constraints['flavor_profile']:
            try:
                flavor_idx = list(self.flavor_encoder.classes_).index(constraints['flavor_profile'])
                flavor_vector = np.eye(flavor_dim)[flavor_idx]
                input_features[current_pos:current_pos + flavor_dim] = flavor_vector * 2.5
            except ValueError:
                print(f"Warning: Unknown flavor profile {constraints['flavor_profile']}")
        current_pos += flavor_dim
        
        # Add region feature
        region_dim = len(self.region_encoder.classes_)
        if 'region' in constraints and constraints['region']:
            try:
                region_idx = list(self.region_encoder.classes_).index(constraints['region'])
                region_vector = np.eye(region_dim)[region_idx]
                input_features[current_pos:current_pos + region_dim] = region_vector * 2.0
            except ValueError:
                print(f"Warning: Unknown region {constraints['region']}")
        current_pos += region_dim
        
        # Add time features
        if 'prep_time' in constraints:
            prep_time = np.array([[constraints['prep_time']]])
            scaled_prep_time = self.time_scaler.transform(prep_time)[0]
            input_features[current_pos] = scaled_prep_time[0] * 1.5
            current_pos += 1
        
        if 'cook_time' in constraints:
            cook_time = np.array([[constraints['cook_time']]])
            scaled_cook_time = self.time_scaler.transform(cook_time)[0]
            input_features[current_pos] = scaled_cook_time[0] * 1.0
        
        # Find nearest neighbors from filtered recipes
        filtered_features = self.features[filtered_df.index]
        model = NearestNeighbors(n_neighbors=min(5, len(filtered_df)), algorithm='auto', metric='cosine')
        model.fit(filtered_features)
        
        distances, indices = model.kneighbors([input_features])
        
        # Get recommended recipes
        recommendations = []
        for dist, idx in zip(distances[0], indices[0]):
            recipe = filtered_df.iloc[idx]
            recommendations.append({
                'name': recipe['name'],
                'ingredients': recipe['ingredients'],
                'diet': recipe['diet'],
                'prep_time': recipe['prep_time'],
                'cook_time': recipe['cook_time'],
                'flavor_profile': recipe['flavor_profile'],
                'course': recipe['course'],
                'region': recipe['region'],
                'source_link': recipe['recipe_link'],
                'ingredient_match': recipe.get('ingredient_match', 0)
            })
        
        return recommendations

def get_user_constraints(generator):
    """Get recipe constraints from user input"""
    print("\nPlease enter your recipe preferences:")
    
    # Get available meal types
    available_meal_types = generator.get_available_meal_types()
    print("\nAvailable meal types:")
    for i, meal_type in enumerate(available_meal_types, 1):
        print(f"{i}. {meal_type}")
    
    # Get user inputs
    constraints = {}
    
    # Meal type
    while True:
        try:
            meal_choice = int(input("\nSelect meal type (enter number): "))
            if 1 <= meal_choice <= len(available_meal_types):
                constraints['course'] = available_meal_types[meal_choice - 1]
                break
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Preparation time
    while True:
        try:
            prep_time = int(input("\nEnter desired preparation time in minutes: "))
            if prep_time > 0:
                constraints['prep_time'] = prep_time
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Cooking time
    while True:
        try:
            cook_time = int(input("Enter desired cooking time in minutes: "))
            if cook_time > 0:
                constraints['cook_time'] = cook_time
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Diet type
    print("\nAvailable diet types:")
    print("1. Vegetarian")
    print("2. Non-vegetarian")
    while True:
        try:
            diet_choice = int(input("Select diet type (1 or 2): "))
            if diet_choice in [1, 2]:
                constraints['diet'] = 'vegetarian' if diet_choice == 1 else 'non-vegetarian'
                break
            else:
                print("Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Flavor profile
    print("\nAvailable flavor profiles:")
    print("1. Sweet")
    print("2. Spicy")
    print("3. Sour")
    print("4. Bitter")
    while True:
        try:
            flavor_choice = int(input("Select flavor profile (1-4): "))
            if 1 <= flavor_choice <= 4:
                flavors = ['sweet', 'spicy', 'sour', 'bitter']
                constraints['flavor_profile'] = flavors[flavor_choice - 1]
                break
            else:
                print("Please enter a number between 1 and 4.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Region
    print("\nAvailable regions:")
    print("1. North Indian")
    print("2. South Indian")
    print("3. East Indian")
    print("4. West Indian")
    print("5. Any Region")
    while True:
        try:
            region_choice = int(input("Select region (1-5): "))
            if 1 <= region_choice <= 5:
                if region_choice == 5:
                    constraints['region'] = None  # Any region
                else:
                    regions = ['North', 'South', 'East', 'West']
                    constraints['region'] = regions[region_choice - 1]
                break
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Ingredients
    print("\nEnter your available ingredients (comma-separated):")
    ingredients = input("Example: milk, sugar, ghee, flour\n").strip()
    constraints['ingredients'] = [ing.strip() for ing in ingredients.split(',') if ing.strip()]
    
    return constraints

def main():
    print("Starting recipe generator...")  # Debug print
    
    # Initialize the recipe generator
    print("Loading dataset...")  # Debug print
    generator = RecipeGenerator('dataset/IndianFoodDatasetCSV.csv')
    print("Dataset loaded successfully!")  # Debug print
    
    while True:
        # Get constraints from user
        constraints = get_user_constraints(generator)
        
        # Generate recommendations
        print("\nGenerating recommendations...")  # Debug print
        recommendations = generator.generate_recipe(constraints)
        
        # Print recommendations
        print("\nRecommended recipes based on your preferences:")
        for i, recipe in enumerate(recommendations, 1):
            print(f"\n{i}. {recipe['name']}")
            print(f"   Diet: {recipe['diet']}")
            print(f"   Meal Type: {recipe['course']}")
            print(f"   Preparation Time: {recipe['prep_time']} minutes")
            print(f"   Cooking Time: {recipe['cook_time']} minutes")
            print(f"   Region: {recipe['region']}")
            print(f"   Ingredients: {recipe['ingredients']}")
            print(f"   Source Link: {recipe['source_link']}")
        
        # Ask if user wants to continue
        choice = input("\nWould you like to generate more recipes? (yes/no): ").lower()
        if choice in ['no', 'n']:
            print("Thank you for using the recipe generator!")
            break
        elif choice not in ['yes', 'y']:
            print("Invalid choice. Please enter 'yes' or 'no'.")

if __name__ == "__main__":
    print("Recipe Generator starting...")  # Debug print
    main() 