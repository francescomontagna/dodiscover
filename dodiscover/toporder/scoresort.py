from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from dodiscover.toporder._base import SteinMixin
from dodiscover.toporder.score import SCORE
from dodiscover.toporder.utils import full_dag, pns


class Scoresort(SCORE):
    """The Score-sort algorithm for causal discovery.

    Score-sort iteratively defines a topological ordering finding the index
    of leaf nodes as the `argmin` of the variance terms of entries of the score vector.
    Then it prunes the fully connected DAG representation of
    the ordering with CAM pruning :footcite:`Buhlmann2013`.
    The method assumes Additive Noise Model and Gaussianity of the noise terms.

    Parameters
    ----------
    eta_G: float, optional
        Regularization parameter for Stein gradient estimator, default is 0.001.
    eta_H : float, optional
        Regularization parameter for Stein Hessian estimator, default is 0.001.
    alpha : float, optional
        Alpha cutoff value for variable selection with hypothesis testing over regression
        coefficients, default is 0.05.
    prune : bool, optional
        If True (default), apply CAM-pruning after finding the topological order.
    n_splines : int, optional
        Number of splines to use for the feature function, default is 10.
        Automatically decreased in case of insufficient samples
    splines_degree: int, optional
        Order of spline to use for the feature function, default is 3.
    estimate_variance : bool, optional
        If True, store estimates the variance of the noise terms from the diagonal of
        Stein Hessian estimator. Default is False.
    pns : bool, optional
        If True, perform Preliminary Neighbour Search (PNS) before CAM pruning step,
        default is False. Allows scaling CAM pruning and ordering to large graphs.
    pns_num_neighbors: int, optional
        Number of neighbors to use for PNS. If None (default) use all variables.
    pns_threshold: float, optional
        Threshold to use for PNS, default is 1.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    Prior knowledge about the included and excluded directed edges in the output DAG
    is supported. It is not possible to provide explicit constraints on the relative
    positions of nodes in the topological ordering. However, explicitly including a
    directed edge in the DAG defines an implicit constraint on the relative position
    of the nodes in the topological ordering (i.e. if directed edge `(i,j)` is
    encoded in the graph, node `i` will precede node `j` in the output order).
    """

    def __init__(
        self,
        eta_G: float = 0.001,
        alpha: float = 0.05,
        prune: bool = True,
        n_splines: int = 10,
        splines_degree: int = 3,
        pns: bool = False,
        pns_num_neighbors: Optional[int] = None,
        pns_threshold: float = 1,
    ):
        super().__init__(
            eta_G, 0.001, alpha, prune, n_splines, splines_degree, False,\
            pns, pns_num_neighbors, pns_threshold
            
        )


    def _find_leaf_iteration(
        self, stein_instance: SteinMixin, X: NDArray, active_nodes: List[int], order: List[int]
    ) -> NDArray:
        """Find a leaf by inspection of the diagonal elements of the Hessian of the log-likelihood.

        Leaf node equals the 'argmin' of the variance of the diagonal terms of the Hessian of
        the log-likelihood.
        After a leaf is identified, it is added to the topological order, and the list of nodes
        without position in the ordering is updated.

        Parameters
        ----------
        stein_instance : SteinMixin
            Instance of the Stein estimator with helper methods.
        X : np.ndarray
            Matrix of the data (active nodes only).
        active_nodes : List[int]
            List of the nodes without a position in the topological ordering.
        order : List[int]
            List of partial order.

        Returns
        -------
        X : np.ndarray
            Matrix of the data without the column corresponding to the identified leaf node.
        """
        score = stein_instance.score(X, self.eta_G)
        leaf = int(score.var(axis=0).argmin())
        leaf = self._get_leaf(
            leaf, active_nodes, order
        )  # get leaf consistent with order imposed by included_edges from self.context
        order.append(active_nodes[leaf])
        active_nodes.pop(leaf)
        X = np.hstack([X[:, 0:leaf], X[:, leaf + 1 :]])
        return X
