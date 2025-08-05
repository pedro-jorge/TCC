from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42 

#HYPERPARAMS = ("hidden_layer_sizes", "activation", "solver", "alpha", "learning_rate", "learning_rate_init")
N_LAYERS = 3
LAYER_MAX_SIZE = 20
LAYER_MIN_SIZE = 2
MIN_DEPTH = 1
MAX_DEPTH = 5

HYPERPARAMS = dict(
    hidden_layer_sizes=[20, 15, 10, 5, 2],
    activation=["relu", "tanh", "logistic"],
    solver=["adam", "lbfgs"],
    alpha=[0.0, 0.1],
    learning_rate=["constant", "invscaling", "adaptive"],
    learning_rate_init=[10e-6, 0.1],
    batch_size=[1, 600],
    beta1=[0.0, 1.0],
    beta2=[0.0, 1.0],
)

hidden_layer_sizes = [0, 0, 0]
activation = ["relu", "tanh", "logistic"]
solver = ["adam", "lbfgs"]
alpha = [0.0, 0.1]
learning_rate = ["constant", "invscaling", "adaptive"]
learning_rate_init = [0.0, 0.1]
batch_size = [1, 600]
beta1 = [0.0, 1.0]
beta2 = [0.0, 1.0]

def get_neural_network(hyperparams, optuna = False):
    if not optuna:
        return make_pipeline(
            #StandardScaler(),
            MLPRegressor(
            hidden_layer_sizes=hyperparams["hidden_layer_sizes"],
            activation=activation[hyperparams["activation"] % len(activation)],
            solver=solver[hyperparams["solver"] % len(solver)],
            alpha=hyperparams["alpha"],
            learning_rate=learning_rate[hyperparams["learning_rate"] % len(learning_rate)],
            learning_rate_init=hyperparams["learning_rate_init"],
            batch_size=hyperparams["batch_size"],
            beta_1=hyperparams["beta1"],
            beta_2=hyperparams["beta2"],
            random_state=RANDOM_STATE,
            )
        )
    
    return make_pipeline(
        #StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hyperparams["hidden_layer_sizes"],
            activation=hyperparams["activation"],
            solver=hyperparams["solver"],
            alpha=hyperparams["alpha"],
            learning_rate=hyperparams["learning_rate"],
            learning_rate_init=hyperparams["learning_rate_init"],
            batch_size=hyperparams["batch_size"],
            beta_1=hyperparams["beta1"],
            beta_2=hyperparams["beta2"],
            random_state=RANDOM_STATE,
        )
    )


def f(X, y, scoring, hyperparams):
    clf = get_neural_network(hyperparams)
    score = np.mean(cross_val_score(clf, X, y, cv=5, scoring=scoring))
    
    return score