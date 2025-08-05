import numpy as np
from numpy.random import randint, choice, uniform
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import optuna 
from pymoo.core.problem import Problem, ElementwiseProblem
import random

from nn_scikit import * 

optuna.logging.set_verbosity(False)


def f(X, y, scoring, hyperparams):
    clf = get_neural_network(hyperparams)
    score = np.mean(cross_val_score(clf, X, y, cv=5, scoring=scoring))
    
    return score


class HyperparameterOptimizationProblem(ElementwiseProblem):
    def __init__(self, X, y, scoring):
        super().__init__(
            n_var=8, 
            n_obj=1, 
            n_ieq_constr=0, 
            xl=[0, 0, 0, 0, 0.0001, 1, 0, 0], 
            xu=[2, 1, 0.1, 2, 0.1, 600, 1, 1]
        )
        
        self.X = X
        self.y = y
        
        self.scoring = scoring

    def _evaluate(self, x, out, *args, **kwargs):
        hyperparams = dict(
            hidden_layer_sizes=HYPERPARAMS["hidden_layer_sizes"],
            activation=int(x[0]),
            solver=int(x[1]),
            alpha=x[2],
            learning_rate=int(x[3]),
            learning_rate_init=x[4],
            batch_size=int(x[5]),
            beta1=x[6],
            beta2=x[7]
        )

        model = get_neural_network(hyperparams)
        score = f(self.X, self.y, self.scoring, hyperparams)
        #np.mean(cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring))

        out["F"] = -score  # Objective function value (maximize R2 score)
        
        
class OptunaMLP:
    def __init__(self, X, y, scoring, n_trials):
        self.X = X
        self.y = y
        self.scoring = scoring
        self.n_trials = n_trials
        self.study = optuna.create_study(direction="minimize")
        self.best_individuals = []
        
    def callback(self, study, trial):
        self.best_individuals.append(self.study.best_value)

    def run(self):
        self.study.optimize(self.objective, n_trials=self.n_trials, callbacks=[self.callback])
        return self.study.best_trial

    def objective(self, trial):
        hidden_layer_sizes_sugg = trial.suggest_categorical("hidden_layer_sizes", [(20, 15, 10, 5, 2), (15, 10, 5, 2), (10, 5, 2)])
        activation_sugg = trial.suggest_categorical("activation", HYPERPARAMS["activation"])
        solver_sugg = trial.suggest_categorical("solver", HYPERPARAMS["solver"])
        alpha_sugg = trial.suggest_float("alpha", HYPERPARAMS["alpha"][0], HYPERPARAMS["alpha"][1])
        learning_rate_sugg = trial.suggest_categorical("learning_rate", HYPERPARAMS["learning_rate"])
        learning_rate_init_sugg = trial.suggest_float("learning_rate_init", HYPERPARAMS["learning_rate_init"][0], HYPERPARAMS["learning_rate_init"][1])
        batch_size_sugg = trial.suggest_int("batch_size", HYPERPARAMS["batch_size"][0], HYPERPARAMS["batch_size"][1])
        beta1_sugg = trial.suggest_float("beta1", HYPERPARAMS["beta1"][0], HYPERPARAMS["beta1"][1])
        beta2_sugg = trial.suggest_float("beta2", HYPERPARAMS["beta2"][0], HYPERPARAMS["beta2"][1])
        
        hyperparam_sugg = dict(
            hidden_layer_sizes=hidden_layer_sizes_sugg,
            activation=activation_sugg,
            solver=solver_sugg,
            alpha=alpha_sugg,
            learning_rate=learning_rate_sugg,
            learning_rate_init=learning_rate_init_sugg,
            batch_size=batch_size_sugg,
            beta1=beta1_sugg,
            beta2=beta2_sugg,
        )
        
        clf = get_neural_network(hyperparam_sugg, optuna=True)
        
        score = np.mean(cross_val_score(clf, self.X, self.y, cv=5, scoring=self.scoring))
        
        return -score


class GeneticAlgorithm:
    def __init__(
        self,
        pop_size: int,
        crossover_rate: float,
        mutation_rate: float,
        n_generations: int,
        scoring: str
    ):
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.population = self.generate_population()
        
        self.scoring = scoring 
        
        self.best_score = np.inf
        self.best_individual = None
        
        self.bests_scores = []
        
    
    def choice_hyperparam(self, hyperparam: str):
        if hyperparam == "hidden_layer_sizes":
            n_layers = randint(MIN_DEPTH, MAX_DEPTH + 1)
            return tuple(sorted(np.random.randint(LAYER_MIN_SIZE, LAYER_MAX_SIZE, size=n_layers)))
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
        elif hyperparam == "batch_size":
            return randint(batch_size[0], batch_size[1] + 1)
        elif hyperparam == "beta1":
            return uniform(beta1[0], beta1[1])
        elif hyperparam == "beta2":
            return uniform(beta2[0], beta2[1])
    
    
    def generate_individual(self):
        individual = dict()
        for hyperparam in HYPERPARAMS.keys():
            individual[hyperparam] = self.choice_hyperparam(hyperparam)
        
        return individual


    def generate_population(self):
        return [self.generate_individual() for _ in range(self.pop_size)]
    
    
    def evaluate_individual(self, individual, X, y):
        return -f(X, y, self.scoring, individual)

    def evaluate_population(self, X, y):
        scores = np.zeros(len(self.population))
        for idx, individual in enumerate(self.population):
            scores[idx] = self.evaluate_individual(individual, X, y)

        #return scores[:self.pop_size], self.population[:self.pop_size]
        scores, individuals = zip(*sorted(zip(scores, self.population), key=lambda x: x[0]))
        self.population = list(individuals[:self.pop_size])
        return scores[:self.pop_size], individuals[:self.pop_size]
        
        # get random idxs from scores
        
    
    
    def crossover(self, parent1, parent2):
        child = {}
        for key in parent1.keys():
            if uniform(0, 1) < self.crossover_rate:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child
    
    
    def mutate(self, individual):
        for key in individual.keys():
            if uniform(0, 1) < self.mutation_rate:
                individual[key] = self.choice_hyperparam(key)
        return individual
    
    
    def run(self, X, y):
        for generation in range(self.n_generations):
            scores, evaluated_population = self.evaluate_population(X, y)
            new_population = []
            
            if scores[0] < self.best_score:
                self.best_score = scores[0]
                self.best_individual = evaluated_population[0].copy()
            
            self.bests_scores.append(self.best_score)
            
            print(f"Generation {generation + 1}/{self.n_generations} -- Best score: {self.best_score}")

            idx = 0
            while len(new_population) < self.pop_size-1:
                if idx >= len(self.population) - 1:
                    break
                # print(f"{idx} ----- {len(self.population)}")
                parent1, parent2 = choice(self.population, size=2, replace=False)
                
                if uniform(0, 1) < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                    new_population.append(child)
                
                if uniform(0, 1) < self.mutation_rate:
                    mutated = random.choice(self.population).copy()
                    mutated = self.mutate(mutated)
                    new_population.append(mutated)
                
                idx += 1
            
            self.population = self.population + new_population
            # print(scores)
        
        # return self.best_individual, make_pipeline(
        #     #StandardScaler(),
        #     MLPRegressor(
        #     hidden_layer_sizes=self.best_individual["hidden_layer_sizes"],
        #     activation=self.best_individual["activation"],
        #     solver=self.best_individual["solver"],
        #     alpha=self.best_individual["alpha"],
        #     learning_rate=self.best_individual["learning_rate"],
        #     learning_rate_init=self.best_individual["learning_rate_init"],
        #     random_state=RANDOM_STATE,
        #     )
        # )
    

class DifferentialEvolution:
    def __init__(
        self, pop_size: int, 
        mutation_rate: float, 
        crossover_rate: float, 
        n_generations: int,
        scoring: str
    ):
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.population = self.generate_population()
        
        self.scoring = scoring 
        
        self.best_score = np.inf
        self.best_individual = None
        
        self.best_scores = []
        
    
    def generate_individual(self):
        individual = dict()
        for hyperparam in HYPERPARAMS.keys():
            individual[hyperparam] = self.choice_hyperparam(hyperparam)
        
        return individual


    def generate_population(self):
        return [self.generate_individual() for _ in range(self.pop_size)]
    

    def choice_hyperparam(self, hyperparam: str):
        if hyperparam == "hidden_layer_sizes":
            n_layers = randint(MIN_DEPTH, MAX_DEPTH + 1)
            return tuple(sorted(np.random.randint(LAYER_MIN_SIZE, LAYER_MAX_SIZE, size=n_layers)))
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
        elif hyperparam == "batch_size":
            return randint(batch_size[0], batch_size[1] + 1)
        elif hyperparam == "beta1":
            return uniform(beta1[0], beta1[1])
        elif hyperparam == "beta2":
            return uniform(beta2[0], beta2[1])
    
    
    def evaluate_individual(self, individual, X, y):
        return -f(X, y, self.scoring, individual)

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
                parent1[key] = parent2[key]
        return parent1
    
    
    def get_individuals(self, idx: int, n: int):
        idxs = [i for i in range(self.pop_size) if i != idx]
        idxs = choice(idxs, size=n, replace=False)
        return [self.population[i] for i in idxs]


    def mutate(self, individual: dict, idx: int, strategy: str = "current-to-best1"):
        if strategy == "current-to-best1":
            r1, r2 = self.get_individuals(idx, 2)
            for hyperparam in HYPERPARAMS.keys():
                if uniform(0, 1) < self.mutation_rate:
                    if type(individual[hyperparam]) == tuple:
                        individual[hyperparam] = random.choice([self.choice_hyperparam(hyperparam), self.best_individual[hyperparam]])
                        
                    elif type(individual[hyperparam]) == int or type(individual[hyperparam]) == np.int64:
                        individual[hyperparam] = random.choice([self.best_individual[hyperparam], r1[hyperparam], r2[hyperparam]])
                        individual[hyperparam] = int(individual[hyperparam])
                    
                    else:
                        individual[hyperparam] = individual[hyperparam] + self.mutation_rate * (self.best_individual[hyperparam] - r1[hyperparam]) * (r1[hyperparam] - r2[hyperparam])
                        individual[hyperparam] = np.abs(individual[hyperparam])
                        
                        if individual[hyperparam] >= HYPERPARAMS[hyperparam][1]:
                            individual[hyperparam] = HYPERPARAMS[hyperparam][0]
                    
            return individual
    
    
    def run(self, X, y):
        for generation in range(self.n_generations):
            scores, evaluated_population = self.evaluate_population(X, y)
            new_population = []

            if scores[0] < self.best_score:
                self.best_score = scores[0]
                self.best_individual = evaluated_population[0].copy()
            
            self.best_scores.append(self.best_score)
            
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