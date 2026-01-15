import numpy as np
import pandas as pd
import streamlit as st

# -------------------- Fixed Parameters (From Question) --------------------
POPULATION_SIZE = 300
CHROMOSOME_LENGTH = 80
GENERATIONS = 50
FITNESS_PEAK = 40
MAX_FITNESS = 80
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.9
ELITISM = 2

# -------------------- Fitness Function --------------------
def fitness_function(x: np.ndarray) -> float:
    ones = np.sum(x)
    return MAX_FITNESS - abs(ones - FITNESS_PEAK)

# -------------------- GA Functions --------------------
def init_population():
    return np.random.randint(0, 2, size=(POPULATION_SIZE, CHROMOSOME_LENGTH))

def tournament_selection(fitness, k=3):
    idx = np.random.choice(len(fitness), k)
    return idx[np.argmax(fitness[idx])]

def one_point_crossover(a, b):
    point = np.random.randint(1, CHROMOSOME_LENGTH)
    return (
        np.concatenate([a[:point], b[point:]]),
        np.concatenate([b[:point], a[point:]])
    )

def mutation(x):
    for i in range(len(x)):
        if np.random.rand() < MUTATION_RATE:
            x[i] = 1 - x[i]
    return x

# -------------------- Run GA --------------------
def run_ga():
    population = init_population()
    history = []

    for gen in range(GENERATIONS):
        fitness = np.array([fitness_function(ind) for ind in population])

        best = fitness.max()
        avg = fitness.mean()
        history.append((best, avg))

        # Elitism
        elite_idx = np.argsort(fitness)[-ELITISM:]
        elites = population[elite_idx]

        new_population = []

        while len(new_population) < POPULATION_SIZE - ELITISM:
            p1 = population[tournament_selection(fitness)]
            p2 = population[tournament_selection(fitness)]

            if np.random.rand() < CROSSOVER_RATE:
                c1, c2 = one_point_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            new_population.append(mutation(c1))
            if len(new_population) < POPULATION_SIZE - ELITISM:
                new_population.append(mutation(c2))

        population = np.vstack([new_population, elites])

    final_fitness = np.array([fitness_function(ind) for ind in population])
    best_idx = np.argmax(final_fitness)

    return population[best_idx], final_fitness[best_idx], history

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Genetic Algorithm Bit Pattern", layout="wide")
st.title("Genetic Algorithm – Bit Pattern Generator")

if st.button("Run Genetic Algorithm"):
    best, best_fitness, history = run_ga()

    st.subheader("Fitness Over Generations")
    df = pd.DataFrame(history, columns=["Best Fitness", "Average Fitness"])
    st.line_chart(df)

    st.subheader("Best Bit Pattern")
    bitstring = ''.join(map(str, best.astype(int)))
    st.code(bitstring)
    st.write(f"Number of ones: {np.sum(best)}")
    st.write(f"Best fitness: {best_fitness}")
