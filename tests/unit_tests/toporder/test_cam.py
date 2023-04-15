import networkx as nx
import numpy as np
import pytest

from dodiscover import make_context
from dodiscover.metrics import structure_hamming_dist, toporder_divergence
from dodiscover.toporder.cam import CAM
from dodiscover.toporder.utils import (
    dummy_dense,
    dummy_groundtruth,
    dummy_sample,
    full_dag,
    orders_consistency,
)


@pytest.fixture
def seed():
    return 42


# -------------------- Unit Tests -------------------- #
def test_given_dataset_when_fitting_CAM_then_shd_larger_equal_dtop(seed):
    X = dummy_sample(seed=seed)
    G = dummy_groundtruth()
    model = CAM()
    context = make_context().variables(observed=X.columns).build()
    model.fit(X, context)
    G_pred = model.graph_
    order_pred = model.order_

    shd = structure_hamming_dist(
        true_graph=G,
        pred_graph=G_pred,
        double_for_anticausal=False,
    )
    d_top = toporder_divergence(G, order_pred)
    assert shd >= d_top


def test_given_dag_and_dag_without_leaf_when_fitting_then_order_estimate_is_consistent(
    seed,
):
    X = dummy_sample(seed=seed)
    order_gt = [2, 1, 3, 0]
    model = CAM()
    context = make_context().variables(observed=X.columns).build()
    model.fit(X, context)
    order_full = model.order_

    pruned_dummy_sample = X[order_gt[:-1]]
    pruned_context = make_context().variables(observed=pruned_dummy_sample.columns).build()
    model.fit(pruned_dummy_sample, pruned_context)
    order_noleaf = model.order_
    assert orders_consistency(order_full, order_noleaf)


def test_given_dataset_and_rescaled_dataset_when_fitting_then_returns_equal_output(seed):
    X = dummy_sample(seed=seed)
    model = CAM()
    context = make_context().variables(observed=X.columns).build()
    model.fit(X, context)
    A = nx.to_numpy_array(model.graph_)
    model.fit(X * 2, context)
    A_rescaled = nx.to_numpy_array(model.graph_)
    assert np.allclose(A, A_rescaled)


def test_given_dataset_and_dataset_with_permuted_column_when_fitting_then_return_consistent_outputs(
    seed,
):
    X = dummy_sample(seed=seed)
    model = CAM()
    context = make_context().variables(observed=X.columns).build()

    # permute sample columns
    permutation = [1, 3, 0, 2]
    permuted_sample = X[permutation]  # permute pd.DataFrame columns

    # Run inference on original and permuted data
    model.fit(permuted_sample, context)
    A_permuted = nx.to_numpy_array(model.graph_)
    order_permuted = model.order_
    model.fit(X, context)
    A = nx.to_numpy_array(model.graph_)
    order = model.order_

    # Match variables order
    back_permutation = [2, 0, 3, 1]
    A_permuted = A_permuted[:, back_permutation]
    A_permuted = A_permuted[back_permutation, :]

    # permutation_order with correct variables name
    permutation_dict = {k: p for k, p in enumerate(permutation)}
    order_permuted = [permutation_dict[o] for o in order_permuted]
    assert order_permuted == order
    assert np.allclose(A_permuted, A)


def test_given_adjacency_when_pruning_then_returns_dag_with_context_included_edges(seed):
    X = dummy_sample(seed=seed)
    model = CAM()
    context = make_context().variables(observed=X.columns).build()
    model.fit(X, context)
    A = nx.to_numpy_array(model.graph_)
    order = model.order_
    A_dense = full_dag(order)
    d = len(X.columns)
    edges = []  # include all edges in A_dense and not in A
    for i in range(d):
        for j in range(d):
            if A_dense[i, j] == 1 and A[i, j] == 0:
                edges.append((i, j))
    included_edges = nx.empty_graph(len(X.columns), create_using=nx.DiGraph)
    included_edges.add_edges_from(edges)
    context = make_context(context).edges(include=included_edges).build()
    model.fit(X, context)
    A_included = nx.to_numpy_array(model.graph_)
    assert np.allclose(A_dense, A_included)


def test_given_adjacency_when_pruning_with_pns_then_returns_dag_with_context_included_edges(seed):
    X = dummy_sample(seed=seed)
    model = CAM(pns=True)
    G_dense = dummy_dense()
    context_builder = make_context()
    context = context_builder.variables(observed=X.columns).edges(include=G_dense).build()
    model.fit(X, context)
    A_included = nx.to_numpy_array(model.graph_)
    A_dense = nx.to_numpy_array(G_dense)
    assert np.allclose(A_dense, A_included)
