from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import random

app = Flask(__name__)

# Define acceptable ranges for each feature
FEATURE_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (50, 300),
    "BloodPressure": (40, 200),
    "SkinThickness": (0, 99),
    "Insulin": (0, 900),
    "BMI": (10, 70),
    "DiabetesPedigreeFunction": (0.0, 2.5),
    "Age": (1, 120),
}

# Genetic Algorithm for feature selection
class GeneticAlgorithm:
    def __init__(self, X, y, population_size=10, generations=10):
        self.X = X
        self.y = y
        self.population_size = population_size
        self.generations = generations
        self.evolution_matrix = []
    
    def initialize_population(self):
        return [np.random.choice([0, 1], size=self.X.shape[1]) for _ in range(self.population_size)]
    
    def fitness(self, individual):
        selected_features = np.where(individual == 1)[0]
        if len(selected_features) == 0:
            return 0
        X_selected = self.X[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, self.y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return accuracy_score(y_test, predictions)
    
    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        return np.concatenate([parent1[:point], parent2[point:]])
    
    def mutate(self, individual):
        mutation_point = random.randint(0, len(individual) - 1)
        individual[mutation_point] = 1 - individual[mutation_point]
        return individual
    
    def run(self):
        population = self.initialize_population()
        for generation in range(self.generations):
            fitness_scores = [self.fitness(individual) for individual in population]
            best_fitness = max(fitness_scores)
            best_individual = population[np.argmax(fitness_scores)]
            
            self.evolution_matrix.append({
                "generation": generation + 1,
                "fitness_scores": fitness_scores,
                "best_individual": best_individual,
                "best_fitness": best_fitness
            })
            
            population = sorted(population, key=self.fitness, reverse=True)[:2]
            while len(population) < self.population_size:
                parent1, parent2 = random.sample(population[:2], 2)
                child = self.crossover(parent1, parent2)
                population.append(self.mutate(child))
        
        best_overall = max(self.evolution_matrix, key=lambda x: x["best_fitness"])
        return np.where(best_overall["best_individual"] == 1)[0]

# Load and preprocess the dataset
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, header=None, names=columns)
    df = shuffle(df, random_state=42)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y, df.columns[:-1]

# Feature engineering and model training
def train_model():
    X, y, feature_names = load_data()
    ga = GeneticAlgorithm(X, y)
    selected_features = ga.run()
    X = X[:, selected_features]
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X = poly.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, scaler, poly, selected_features, feature_names, accuracy

# Train the model
model, scaler, poly, selected_features, feature_names, accuracy = train_model()

@app.route("/")
def index():
    return render_template("index.html", features=feature_names[selected_features], accuracy=accuracy)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Validate user inputs
        data = []
        for feature in feature_names[selected_features]:
            value = float(request.form[feature])
            min_val, max_val = FEATURE_RANGES[feature]
            if not (min_val <= value <= max_val):
                return jsonify({"error": f"{feature} value {value} is out of the realistic range ({min_val}-{max_val})."})
            data.append(value)

        # Proceed with prediction if inputs are valid
        input_data = np.array(data).reshape(1, -1)
        input_data = poly.transform(input_data)
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        return jsonify({"result": result, "probability": probability})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)