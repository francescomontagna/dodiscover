from typing import List, Optional, Tuple, Any

import numpy as np
from numpy.typing import NDArray

from dodiscover.toporder._base import BaseTopOrder
from dodiscover.toporder.utils import full_dag, pns
from dodiscover.ci.kernel_test import KernelCITest


class RESIT(BaseTopOrder):
    """The RESIT algorithm.

    RESIT :footcite:`Peters2014resit` algorithm iteratively find leaf nodes of a
    causal graph by performing regression with subsequent independence test.

    Parameters
    ----------
    regressor : Any
        regressor object implementing 'fit' and 'predict' function.
        Regressor to compute residuals. This regressor object must have ``fit`` method
        and ``predict`` function like scikit-learn's model.
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
    pns : bool, optional
        If True, perform Preliminary Neighbour Search (PNS) before CAM pruning step,
        default is None, which activates PNS only for graphs strictly larger than 20 nodes.
        Allows scaling CAM pruning and ordering to large graphs.
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
        regressor : Any = None,
        alpha: float = 0.05,
        prune: bool = True,
        n_splines: int = 10,
        splines_degree: int = 3,
        pns: bool = None,
        pns_num_neighbors: Optional[int] = None,
        pns_threshold: float = 1,
    ):
        super().__init__(
            alpha, prune, n_splines, splines_degree, pns, pns_num_neighbors, pns_threshold
        )

        # Check parameters
        if regressor is None:
            raise ValueError("Specify regression model in 'regressor'.")
        else:
            if not (hasattr(regressor, "fit") and hasattr(regressor, "predict")):
                raise ValueError("'regressor' has no fit or predict method.")
            
        self._regressor = regressor


    def _top_order(self, X: NDArray) -> Tuple[NDArray, List[int]]:
        """Find the topological ordering of the causal variables from X dataset.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_nodes)
            Dataset of observations of the causal variables

        Returns
        -------
        A_dense : np.ndarray
            Fully connected matrix admitted by the topological ordering
        order : List[int]
            Inferred causal order

        Notes
        -----
        Differently from the implementation proposed in RESIT original paper,
        we replace the Hilbert-Schmidt Independence Criterion (HSIC) test
        of independence :footcite:`Gretton2007hsic` with the Kernel-based
        Conditional Independence Test :footcite:`Zhang2011`.

        References
        ----------
        .. footbibliography::
        """
        
        active_nodes = np.arange(X.shape[1])
        order = []
        kernel_indep_test = KernelCITest()
        for _ in range(X.shape[1]):
            if len(active_nodes) == 1:
                order.insert(0, active_nodes[0])
                continue

            kci_stats = []
            for k in active_nodes:
                # Regress Xk on {Xi}
                predictors = [i for i in active_nodes if i != k]
                self._regressor.fit(X[:, predictors], X[:, k])
                residual = X[:, k] - self._regressor.predict(X[:, predictors])
                # Measure dependence between residuals and {Xi}
                kci_stat, _ = kernel_indep_test(residual, X[:, predictors])
                kci_stats.append(kci_stat)

            # Add leaf to the order and remove it from the active_nodes
            leaf = active_nodes[np.argmin(kci_stats)]
            active_nodes = active_nodes[active_nodes != leaf]
            order.insert(0, leaf)

        return full_dag(order), order
    

    def _prune(self, X: NDArray, A_dense: NDArray) -> NDArray:
        """Pruning of the fully connected adj. matrix representation of the inferred order.

        If self.do_pns = True or self.do_pns is None and number of nodes >= 20, then
        Preliminary Neighbors Search is applied before CAM pruning.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_nodes)
            Matrix of the data.
        A_dense : np.ndarray of shape (n_nodes, n_nodes)
            Dense adjacency matrix to be pruned.

        Returns
        -------
        A : np.ndarray
            The pruned adjacency matrix output of the causal discovery algorithm.
        """
        d = A_dense.shape[0]
        if (self.do_pns) or (self.do_pns is None and d > 20):
            A_dense = pns(
                A=A_dense,
                X=X,
                pns_threshold=self.pns_threshold,
                pns_num_neighbors=self.pns_num_neighbors,
            )
        return super()._prune(X, A_dense)

