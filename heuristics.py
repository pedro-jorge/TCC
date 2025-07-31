import numpy as np
from numpy.random import randint, choice, uniform
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import optuna 
from pymoo.core.problem import Problem, ElementwiseProblem

#optuna.logging.set_verbosity(False)

RANDOM_STATE = 42

# if hyperparam == "hidden_layer_sizes":
#     return tuple(sorted(np.random.randint(1, 50, size=np.random.randint(1, 4))))

HYPERPARAMS = ("activation", "solver", "alpha", "learning_rate", "learning_rate_init")
N_LAYERS = 3
LAYER_MAX_SIZE = 20
LAYER_MIN_SIZE = 2

hidden_layer_sizes = [0, 0, 0]
activation = ["relu", "tanh", "logistic"]
solver = ["adam", "lbfgs"]
alpha = [0.0, 0.1]
learning_rate = ["constant", "invscaling", "adaptive"]
learning_rate_init = [0.0001, 0.1]


def get_neural_network(hyperparams):
    return make_pipeline(
        #StandardScaler(),
        MLPRegressor(
        hidden_layer_sizes=hyperparams["hidden_layer_sizes"],
        activation=activation[hyperparams["activation"] % len(activation)],
        solver=solver[hyperparams["solver"] % len(solver)],
        alpha=hyperparams["alpha"],
        learning_rate=learning_rate[hyperparams["learning_rate"] % len(learning_rate)],
        learning_rate_init=hyperparams["learning_rate_init"],
        random_state=RANDOM_STATE,
        )
    )


def f(X, y, hyperparams):
    clf = get_neural_network(hyperparams)
    return np.mean(cross_val_score(clf, X, y, cv=5, scoring="r2"))


class HyperparameterOptimizationProblem(ElementwiseProblem):
    def __init__(self, X, y):
        super().__init__(
            n_var=5, 
            n_obj=1, 
            n_ieq_constr=0, 
            xl=[0, 0, 0, 0, 0.0001], 
            xu=[2, 1, 0.1, 2, 0.1]
        )
        
        self.X = X
        self.y = y

    def _evaluate(self, x, out, *args, **kwargs):
        hyperparams = dict(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=int(x[0]),
            solver=int(x[1]),
            alpha=x[2],
            learning_rate=int(x[3]),
            learning_rate_init=x[4]
        )

        model = get_neural_network(hyperparams)
        score = cross_val_score(model, self.X, self.y, cv=5, scoring="r2").mean()

        out["F"] = -score  # Objective function value (maximize R2 score)
        
        
class OptunaMLP:
    def __init__(self, X, y, n_trials):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.study = optuna.create_study(direction="minimize")

    def run(self):
        self.study.optimize(self.objective, n_trials=self.n_trials)
        return self.study.best_trial

    def objective(self, trial):
        hidden_layer_sizes_sugg = trial.suggest_categorical("hidden_layer_sizes", [(20, 15, 10, 5, 2), (15, 10, 5, 2), (10, 5, 2)])
        activation_sugg = trial.suggest_categorical("activation", activation)
        solver_sugg = trial.suggest_categorical("solver", solver)
        alpha_sugg = trial.suggest_float("alpha", alpha[0], alpha[1])
        learning_rate_sugg = trial.suggest_categorical("learning_rate", learning_rate)
        learning_rate_init_sugg = trial.suggest_float("learning_rate_init", learning_rate_init[0], learning_rate_init[1])

        clf = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes_sugg, 
            activation=activation_sugg, solver=solver_sugg, 
            alpha=alpha_sugg, 
            learning_rate=learning_rate_sugg, 
            learning_rate_init=learning_rate_init_sugg,
            random_state=RANDOM_STATE,
        )
        
        score = cross_val_score(clf, self.X, self.y, cv=5, scoring="r2").mean()
        return -score


class GeneticAlgorithm:
    def __init__(
        self,
        pop_size: int,
        crossover_rate: float,
        mutation_rate: float,
        n_generations: int,
    ):
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.population = self.generate_population()
        
        self.best_score = -np.inf
        self.best_individual = None
        
    
    def choice_hyperparam(self, hyperparam: str):
        if hyperparam == "hidden_layer_sizes":
            return tuple(sorted(np.random.randint(1, 50, size=np.random.randint(1, 4))))
        elif hyperparam == "activation":
            return randint(0, len(activation))
        elif hyperparam == "solver":
            return randint(0, len(solver))
        elif hyperparam == "alpha":
            return uniform(alpha[0], alpha[1])
        elif hyperparam == "learning_rate":
            return randint(0, len(learning_rate))
        elif hyperparam == "learning_rate_init":
            return uniform(learning_rate_init[0], learning_rate_init[1])


    def generate_individual(self):
        return dict(
            hidden_layer_sizes=self.choice_hyperparam("hidden_layer_sizes"),
            activation=self.choice_hyperparam("activation"),
            solver=self.choice_hyperparam("solver"),
            alpha=self.choice_hyperparam("alpha"),
            learning_rate=self.choice_hyperparam("learning_rate"),
            learning_rate_init=self.choice_hyperparam("learning_rate_init"),
        )


    def generate_population(self):
        return [self.generate_individual() for _ in range(self.pop_size)]
    
    
    def evaluate_individual(self, individual, X, y):
        return f(X, y, individual)


    def evaluate_population(self, X, y):
        scores = np.zeros(self.pop_size)
        for idx, individual in enumerate(self.population):
            scores[idx] = f(X, y, individual)
            
        scores, individuals = zip(*sorted(zip(scores, self.population), key=lambda x: x[0], reverse=True))
        return scores, individuals
    
    def crossover(self, parent1, parent2):
        child = {}
        for key in parent1.keys():
            if uniform(0, 1) < self.crossover_rate:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child
    
    
    def mutate(self, individual):
        key = choice(list(individual.keys()))
        if key == "hidden_layer_sizes":
            individual[key] = self.choice_hyperparam("hidden_layer_sizes")
        elif key == "activation":
            individual[key] = self.choice_hyperparam("activation")
        elif key == "solver":
            individual[key] = self.choice_hyperparam("solver")
        elif key == "alpha":
            individual[key] = self.choice_hyperparam("alpha")
        elif key == "learning_rate":
            individual[key] = self.choice_hyperparam("learning_rate")
        elif key == "learning_rate_init":
            individual[key] = self.choice_hyperparam("learning_rate_init")
        return individual
    
    
    def run(self, X, y):
        for generation in range(self.n_generations):
            scores, evaluated_population = self.evaluate_population(X, y)
            new_population = []
            
            if scores[0] > self.best_score:
                self.best_score = scores[0]
                self.best_individual = evaluated_population[0].copy()
            
            print(f"Generation {generation + 1}/{self.n_generations} -- Best score: {self.best_score}")
            
            # for i in range(0, self.pop_size-2):
            #     parent1, parent2 = self.population[i], self.population[i + 1]
            #     if uniform(0, 1) < self.crossover_rate:
            #         child = self.crossover(parent1, parent2)
            #     else:
            #         child = parent1.copy()
                
            #     if uniform(0, 1) < self.mutation_rate:
            #         child = self.mutate(child)

            #     if self.evaluate_individual(child, X, y) > scores[i]:
            #         new_population.append(child)
            #     else:
            #         new_population.append(parent1.copy())

            while len(new_population) < self.pop_size:
                parent1, parent2 = choice(evaluated_population, size=2, replace=False)
                if uniform(0, 1) < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                if uniform(0, 1) < self.mutation_rate:
                    child = self.mutate(child)
                    
                new_population.append(child)
            
            self.population = new_population[: self.pop_size]
            print(scores)
        
        return self.best_individual, make_pipeline(
            #StandardScaler(),
            MLPRegressor(
            hidden_layer_sizes=self.best_individual["hidden_layer_sizes"],
            activation=self.best_individual["activation"],
            solver=self.best_individual["solver"],
            alpha=self.best_individual["alpha"],
            learning_rate=self.best_individual["learning_rate"],
            learning_rate_init=self.best_individual["learning_rate_init"],
            random_state=RANDOM_STATE,
            )
        )
    

class DifferentialEvolution:
    def __init__(
        self, pop_size: int, 
        mutation_rate: float, 
        crossover_rate: float, 
        n_generations: int
    ):
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.population = self.generate_population()
        
        self.best_score = np.inf
        self.best_individual = None
        
    
    def generate_individual(self):
        return dict(
            hidden_layer_sizes=self.choice_hyperparam("hidden_layer_sizes"),
            activation=self.choice_hyperparam("activation"),
            solver=self.choice_hyperparam("solver"),
            alpha=self.choice_hyperparam("alpha"),
            learning_rate=self.choice_hyperparam("learning_rate"),
            learning_rate_init=self.choice_hyperparam("learning_rate_init"),
        )


    def generate_population(self):
        return [self.generate_individual() for _ in range(self.pop_size)]
    

    def choice_hyperparam(self, hyperparam: str):
        if hyperparam == "hidden_layer_sizes":
            return tuple(sorted(np.random.randint(LAYER_MIN_SIZE, LAYER_MAX_SIZE, size=N_LAYERS)))
        elif hyperparam == "activation":
            return randint(0, len(activation))
        elif hyperparam == "solver":
            return randint(0, len(solver))
        elif hyperparam == "alpha":
            return uniform(alpha[0], alpha[1])
        elif hyperparam == "learning_rate":
            return randint(0, len(learning_rate))
        elif hyperparam == "learning_rate_init":
            return uniform(learning_rate_init[0], learning_rate_init[1])
    
    
    def evaluate_individual(self, individual, X, y):
        return -f(X, y, individual)

    def evaluate_population(self, X, y):
        scores = np.zeros(self.pop_size)
        for idx, individual in enumerate(self.population):
            try:
                scores[idx] = self.evaluate_individual(individual, X, y)
            except Exception as e:
                print(e)

        scores, individuals = zip(*sorted(zip(scores, self.population), key=lambda x: x[0]))
        return scores, individuals
    
    
    def crossover(self, parent1, parent2):
        for key in parent1.keys():
            if uniform(0, 1) < self.crossover_rate:
                if key == "hidden_layer_sizes":
                    pass
                    # idx_layer = randint(0, N_LAYERS-1)
                    # try:
                    #     layers = list(parent1[key])
                    #     layers[idx_layer] = parent2[key][idx_layer]
                    #     parent1[key] = tuple(sorted(layers))
                    # except Exception as e:
                    #     print(e)
                else:
                    parent1[key] = parent2[key]
        return parent1
    
    
    def get_individuals(self, idx: int, n: int):
        idxs = [i for i in range(self.pop_size) if i != idx]
        idxs = choice(idxs, size=n, replace=False)
        return [self.population[i] for i in idxs]


    def mutate(self, individual: dict, idx: int, strategy: str = "current-to-best1"):
        # if strategy == "rand1":
        #     base, r1, r2 = self.get_individuals(idx, 3)
        #     for hyperparam in HYPERPARAMS:
        #         if type(base[hyperparam]) is int:
        #             base[hyperparam] = base[hyperparam] + self.mutation_rate * (r1[hyperparam] - r2[hyperparam])
        #             base[hyperparam] = int(base[hyperparam])
                    
        #     return base

        if strategy == "current-to-best1":
            r1, r2 = self.get_individuals(idx, 2)
            for hyperparam in HYPERPARAMS:
                if type(individual[hyperparam]) is int:
                    individual[hyperparam] = individual[hyperparam] + self.mutation_rate * (self.best_individual[hyperparam] - r1[hyperparam]) + self.mutation_rate * (r1[hyperparam] - r2[hyperparam])
                    individual[hyperparam] = int(individual[hyperparam])
                    
                # if hyperparam == "hidden_layer_sizes":
                #     layers = []
                #     for idx in range(N_LAYERS):
                #         layers.append(int(individual[hyperparam][idx] + self.mutation_rate * (r1[hyperparam][idx] - r2[hyperparam][idx])))
                #     individual[hyperparam] = tuple(sorted(layers))
                else:
                    individual[hyperparam] = individual[hyperparam] + self.mutation_rate * (self.best_individual[hyperparam] - r1[hyperparam]) * (r1[hyperparam] - r2[hyperparam])
                    individual[hyperparam] = np.abs(individual[hyperparam])
                    
            return individual
    
    
    def run(self, X, y):
        for generation in range(self.n_generations):
            scores, evaluated_population = self.evaluate_population(X, y)
            new_population = []

            if scores[0] < self.best_score:
                self.best_score = scores[0]
                self.best_individual = evaluated_population[0].copy()
            
            print(f"Generation {generation + 1}/{self.n_generations} -- Best score: {self.best_score}")

            for idx in range(self.pop_size):
                mut_vector = self.mutate(self.population[idx], idx)
                rand_vector = self.get_individuals(idx, 1)[0]
                cross_vector = self.crossover(rand_vector, mut_vector)
                
                cross_vector_f = self.evaluate_individual(cross_vector, X, y)
                    
                if cross_vector_f < scores[idx]:
                    new_population.append(cross_vector)
                else:
                    new_population.append(self.population[idx].copy())

            self.population = new_population[: self.pop_size]
            print(scores)
        
        return self.best_individual, make_pipeline(
            #StandardScaler(),
            MLPRegressor(
            hidden_layer_sizes=self.best_individual["hidden_layer_sizes"],
            activation=self.best_individual["activation"],
            solver=self.best_individual["solver"],
            alpha=self.best_individual["alpha"],
            learning_rate=self.best_individual["learning_rate"],
            learning_rate_init=self.best_individual["learning_rate_init"],
            random_state=RANDOM_STATE,
            )
        )