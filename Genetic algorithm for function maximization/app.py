from flask import Flask, render_template, request
import numpy as np
import random

app = Flask(__name__)

# Funkcja do maksymalizacji
def target_function(x):
    return x * np.sin(x)

# Algorytm Genetyczny
def genetic_algorithm(lower_bound, upper_bound, population_size, generations, mutation_rate):
    def fitness(individual):
        return target_function(individual)

    def create_population(size):
        return np.random.uniform(lower_bound, upper_bound, size)

    def select_parents(population, fitnesses):
        tournament_size = 3
        selected = []
        for _ in range(2):
            tournament = random.sample(range(len(population)), tournament_size)
            best = max(tournament, key=lambda idx: fitnesses[idx])
            selected.append(population[best])
        return selected

    def crossover(parent1, parent2):
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2

    def mutate(individual):
        if random.random() < mutation_rate:
            mutation = np.random.uniform(-1, 1)
            individual = np.clip(individual + mutation, lower_bound, upper_bound)
        return individual

    population = create_population(population_size)
    for generation in range(generations):
        fitnesses = np.array([fitness(ind) for ind in population])
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        population = np.array(new_population[:population_size])

    best_idx = np.argmax([fitness(ind) for ind in population])
    return population[best_idx], fitness(population[best_idx])

# Strona główna
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            lower_bound = float(request.form["lower_bound"])
            upper_bound = float(request.form["upper_bound"])
            population_size = int(request.form["population_size"])
            generations = int(request.form["generations"])
            mutation_rate = float(request.form["mutation_rate"])
            
            # Uruchomienie algorytmu genetycznego
            best_x, best_fitness = genetic_algorithm(
                lower_bound, upper_bound, population_size, generations, mutation_rate
            )
            result = {
                "best_x": round(best_x, 4),
                "best_fitness": round(best_fitness, 4),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "population_size": population_size,
                "generations": generations,
                "mutation_rate": mutation_rate
            }
        except ValueError:
            result = {"error": "Podano nieprawidłowe dane. Spróbuj ponownie."}

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
