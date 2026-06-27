"""
Model Multilayer Perceptron (MLP) untuk klasifikasi uplift potential.
"""

from sklearn.neural_network import MLPClassifier


def build_mlp(hidden_layers=(64, 32), max_iter=500, random_state=42, early_stopping=True):
    """
    Buat MLPClassifier dengan arsitektur multilayer perceptron.

    Parameters
    ----------
    hidden_layers : tuple
        Jumlah neuron tiap hidden layer, e.g. (64, 32).
    max_iter : int
        Iterasi maksimum training.
    random_state : int
        Seed reproducibility.
    early_stopping : bool
        Aktifkan validasi awal untuk mencegah overfit.

    Returns
    -------
    MLPClassifier
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        max_iter=max_iter,
        shuffle=True,
        random_state=random_state,
        early_stopping=early_stopping,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False,
    )
