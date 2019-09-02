"""
Implementation of the method proposed in the paper:
'Adversarial Attacks on Graph Neural Networks via Meta Learning'
by Daniel Zügner, Stephan Günnemann
Published at ICLR 2019 in New Orleans, USA.
Copyright (C) 2019
Daniel Zügner
Technical University of Munich
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                    loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                         loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.
    Parameters
    ----------
    adj : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.
    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None,
                                 random_state=None):

    """
    Split the arrays or matrices into random train, validation and test subsets.
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
        Proportion of the dataset included in the train split.
    val_size : float, default 0.3
        Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
        Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
        If not None, data is split in a stratified fashion, using this as the class labels.
    random_state : int or None, default None
        Random_state is the seed used by the random number generator;
    Returns
    -------
    splitting : list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.
    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result


def preprocess_graph(adj):
    """
    Perform the processing of the adjacency matrix proposed by Kipf et al. 2017.

    Parameters
    ----------
    adj: sp.spmatrix
        Input adjacency matrix.

    Returns
    -------
    The matrix (D+1)^(-0.5) (adj + I) (D+1)^(-0.5)

    """
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = adj_.sum(1).A1
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized


def unravel_index_tf(ix, shape):
    """
    Unravels the input index similar to np.unravel_index. That is, given the "flat"
    (i.e. between 0 and shape[0] * shape[1] - 1) input index and a 2D shape computes
    the 2D index corresponding to the input index.

    Parameters
    ----------
    ix: tf.int32
        The input index.
    shape: tuple or list of ints with length 2
        2D shape (e.g. adjacency matrix dimensions).

    Returns
    -------
    tf.Tensor, dtype int, shape (2,)
        The index in the 2D shape corresponding to the "flat" input index ix.

    """
    output_list = []
    output_list.append(ix // (shape[1]))
    output_list.append(ix % (shape[1]))
    return tf.stack(output_list)


def ravel_index(ix, shape):
    """
    "Flattens" the 2D input index into a single index on the flattened matrix, similar to np.ravel_multi_index.

    Parameters
    ----------
    ix: array or list of ints of shape (2,)
        The 2D input index.
    shape: list or tuple of ints of length 2
        The shape of the corresponding matrix.

    Returns
    -------
    int between 0 and shape[0]*shape[1]-1
        The index on the flattened matrix corresponding to the 2D input index.

    """
    return ix[0]*shape[1] + ix[1]


def ravel_multiple_indices(ixs, shape):
    """
    "Flattens" multiple 2D input indices into indices on the flattened matrix, similar to np.ravel_multi_index.
    Does the same as ravel_index but for multiple indices at once.
    Parameters
    ----------
    ixs: array of ints shape (n, 2)
        The array of n indices that will be flattened.

    shape: list or tuple of ints of length 2
        The shape of the corresponding matrix.

    Returns
    -------
    array of n ints between 0 and shape[0]*shape[1]-1
        The indices on the flattened matrix corresponding to the 2D input indices.

    """
    return ixs[:, 0] * shape[1] + ixs[:, 1]


def compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    """
    Computes thelog likelihood of the observed Powerlaw distribution given the Powerlaw exponent alpha.

    Parameters
    ----------
    n: int
        The number of samples in the observed distribution whose value is >= d_min.

    alpha: float
        The Powerlaw exponent for which the log likelihood is to be computed.

    sum_log_degrees: float
        The sum of the logs of samples in the observed distribution whose values are >= d_min.

    d_min: int
        The minimum degree to be considered in the Powerlaw computation.

    Returns
    -------
    float
        The log likelihood of the given observed Powerlaw distribution and exponend alpha.

    """
    return n * tf.log(alpha) + n * alpha * tf.log(d_min) + (alpha + 1) * sum_log_degrees


def update_sum_log_degrees(sum_log_degrees_before, n_old, d_old, d_new, d_min):
    """
    Compute the sum of the logs of samples in the observed distribution whose values are >= d_min for a single edge
    changing in the graph. That is, given that two degrees in the graph change from d_old to d_new respectively
    (resulting from adding or removing a single edge), compute the updated sum of log degrees >= d_min.

    Parameters
    ----------
    sum_log_degrees_before: tf.Tensor of floats of length n
        The sum of log degrees >= d_min before the change.

    n_old: tf.Tensor of ints of length n
        The number of degrees >= d_min before the change.

    d_old: tf.Tensor of ints, shape [n, 2]
        The old (i.e. before change) degrees of the two nodes affected by an edge to be inserted/removed. n corresponds
        to the number of edges for which this will be computed in a vectorized fashion.

    d_new: tf.Tensor of ints, shape [n,2]
        The new (i.e. after the change) degrees of the two nodes affected by an edge to be inserted/removed.
        n corresponds to the number of edges for which this will be computed in a vectorized fashion.

    d_min: int
        The minimum degree considered in the Powerlaw distribution.

    Returns
    -------
    sum_log_degrees_after: tf.Tensor of floats shape (n,)
        The updated sum of log degrees whose values are >= d_min after a potential edge being added/removed.

    new_n: tf.Tensor dtype int shape (n,)
        The updated number of degrees which are >= d_min after a potential edge being added/removed.


    """

    # Find out whether the degrees before and after the change are above the threshold d_min.
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min

    # Mask out the degrees whose values are below d_min by multiplying them by 0.
    d_old_in_range = tf.multiply(d_old, tf.cast(old_in_range, tf.float32))
    d_new_in_range = tf.multiply(d_new, tf.cast(new_in_range, tf.float32))

    # Update the sum by subtracting the old values and then adding the updated logs of the degrees.
    sum_log_degrees_after = sum_log_degrees_before - tf.reduce_sum(tf.log(tf.maximum(d_old_in_range, 1)),
                                                                   axis=1) + tf.reduce_sum(
        tf.log(tf.maximum(d_new_in_range, 1)), axis=1)

    # Update the number of degrees >= d_min
    new_n = tf.cast(n_old, tf.int64) - tf.count_nonzero(old_in_range, axis=1) + tf.count_nonzero(new_in_range, axis=1)

    return sum_log_degrees_after, new_n


def compute_alpha(n, sum_log_degrees, d_min):
    """
    Compute the maximum likelihood value of the Powerlaw exponent alpha of the degree distribution.

    Parameters
    ----------
    n: int
        The number of degrees >= d_min

    sum_log_degrees: float
        The sum of log degrees >= d_min

    d_min: int
        The minimum degree considered in the Powerlaw distribution.

    Returns
    -------
    alpha: float
        The maximum likelihood estimate of the Powerlaw exponent alpha.

    """
    return n / (sum_log_degrees - n * tf.log(d_min - 0.5)) + 1


def degree_sequence_log_likelihood(degree_sequence, d_min):
    """
    Compute the (maximum) log likelihood of the Powerlaw distribution fit on a degree distribution.

    Parameters
    ----------
    degree_sequence: tf.Tensor dtype int shape (N,)
        Observed degree distribution.

    d_min: int
        The minimum degree considered in the Powerlaw distribution.

    Returns
    -------
    ll: tf.Tensor dtype float, (scalar)
        The log likelihood under the maximum likelihood estimate of the Powerlaw exponent alpha.

    alpha: tf.Tensor dtype float (scalar)
        The maximum likelihood estimate of the Powerlaw exponent.

    n: int
        The number of degrees in the degree sequence that are >= d_min.

    sum_log_degrees: tf.Tensor dtype float (scalar)
        The sum of the log of degrees in the distribution which are >= d_min.

    """
    # Determine which degrees are to be considered, i.e. >= d_min.
    in_range = tf.greater_equal(degree_sequence, d_min)
    # Sum the log of the degrees to be considered
    sum_log_degrees = tf.reduce_sum(tf.log(tf.boolean_mask(degree_sequence, in_range)))
    # Number of degrees >= d_min
    n = tf.cast(tf.count_nonzero(in_range), tf.float32)
    # Maximum likelihood estimate of the Powerlaw exponent
    alpha = compute_alpha(n, sum_log_degrees, d_min)
    # Log likelihood under alpha
    ll = compute_log_likelihood(n, alpha, sum_log_degrees, d_min)

    return ll, alpha, n, sum_log_degrees


def updated_log_likelihood_for_edge_changes(node_pairs, adjacency_matrix, d_min):
    """
    Compute the change of the log likelihood of the Powerlaw distribution fit on the input adjacency matrix's degree
    distribution that results when adding/removing edges for the input node pairs. Assumes an undirected unweighted
    graph.

    Parameters
    ----------
    node_pairs: tf.Tensor, shape (e, 2) dtype int
        The e node pairs to consider, where each node pair consists of the two indices of the nodes.

    adjacency_matrix: tf.Tensor shape (N,N) dtype int
        The input adjacency matrix. Assumed to be unweighted and symmetric.

    d_min: int
        The minimum degree considered in the Powerlaw distribution.

    Returns
    -------
    new_ll: tf.Tensor of shape (e,) and dtype float
        The log likelihoods for node pair in node_pairs obtained when adding/removing the edge for that node pair.

    new_alpha: tf.Tensor of shape (e,) and dtype float
        For each node pair, contains the maximum likelihood estimates of the Powerlaw distributions obtained when
        adding/removing the edge for that node pair.

    new_n: tf.Tensor of shape (e,) and dtype float
        The updated number of degrees which are >= d_min for each potential edge being added/removed.

    sum_log_degrees_after: tf.Tensor of floats shape (e,)
        The updated sum of log degrees whose values are >= d_min for each of the e potential edges being added/removed.

    """

    # For each node pair find out whether there is an edge or not in the input adjacency matrix.
    edge_entries_before = tf.cast(tf.gather_nd(adjacency_matrix, tf.cast(node_pairs, tf.int32)), tf.float32)
    # Compute the degree for each node
    degree_seq = tf.reduce_sum(adjacency_matrix, 1)

    # Determine which degrees are to be considered, i.e. >= d_min.
    in_range = tf.greater_equal(degree_seq, d_min)
    # Sum the log of the degrees to be considered
    sum_log_degrees = tf.reduce_sum(tf.log(tf.boolean_mask(degree_seq, in_range)))
    # Number of degrees >= d_min
    n = tf.cast(tf.count_nonzero(in_range), tf.float32)

    # The changes to the edge entries to add an edge if none was present and remove it otherwise.
    # i.e., deltas[ix] = -1 if edge_entries[ix] == 1 else 1
    deltas = -2 * edge_entries_before + 1

    # The degrees of the nodes in the input node pairs
    d_edges_before = tf.gather(degree_seq, tf.cast(node_pairs, tf.int32))
    # The degrees of the nodes in the input node pairs after performing the change (i.e. adding the respective value of
    # delta.
    d_edges_after = tf.gather(degree_seq, tf.cast(node_pairs, tf.int32)) + deltas[:, None]
    # Sum the log of the degrees after the potential changes which are >= d_min
    sum_log_degrees_after, new_n = update_sum_log_degrees(sum_log_degrees, n, d_edges_before, d_edges_after, d_min)
    # Update the number of degrees >= d_min
    new_n = tf.cast(new_n, tf.float32)

    # Updated estimates of the Powerlaw exponents
    new_alpha = compute_alpha(new_n, sum_log_degrees_after, d_min)
    # Updated log likelihood values for the Powerlaw distributions
    new_ll = compute_log_likelihood(new_n, new_alpha, sum_log_degrees_after, d_min)

    return new_ll, new_alpha, new_n, sum_log_degrees_after


def likelihood_ratio_filter(node_pairs, modified_adjacency, original_adjacency, d_min, threshold=0.004):
    """
    Filter the input node pairs based on the likelihood ratio test proposed by Zügner et al. 2018, see
    https://dl.acm.org/citation.cfm?id=3220078. In essence, for each node pair return 1 if adding/removing the edge
    between the two nodes does not violate the unnoticeability constraint, and return 0 otherwise. Assumes unweighted
    and undirected graphs.

    Parameters
    ----------
    node_pairs: tf.Tensor, shape (e, 2) dtype int
        The e node pairs to consider, where each node pair consists of the two indices of the nodes.

    modified_adjacency: tf.Tensor shape (N,N) dtype int
        The input (modified) adjacency matrix. Assumed to be unweighted and symmetric.

    original_adjacency: tf.Tensor shape (N,N) dtype int
        The input (original) adjacency matrix. Assumed to be unweighted and symmetric.

    d_min: int
        The minimum degree considered in the Powerlaw distribution.

    threshold: float, default 0.004
        Cutoff value for the unnoticeability constraint. Smaller means stricter constraint. 0.004 corresponds to a
        p-value of 0.95 in the Chi-square distribution with one degree of freedom.

    Returns
    -------
    allowed_mask: tf.Tensor, shape (e,), dtype bool
        For each node pair p return True if adding/removing the edge p does not violate the
        cutoff value, False otherwise.

    current_ratio: tf.Tensor, shape (), dtype float
        The current value of the log likelihood ratio.

    """

    N = int(modified_adjacency.shape[0])

    original_degree_sequence = tf.cast(tf.reduce_sum(original_adjacency, axis=1), tf.float32)
    current_degree_sequence = tf.cast(tf.reduce_sum(modified_adjacency, axis=1), tf.float32)

    # Concatenate the degree sequences
    concat_degree_sequence = tf.concat((current_degree_sequence[None, :], original_degree_sequence[None, :]), axis=1)
    # Compute the log likelihood values of the original, modified, and combined degree sequences.
    ll_orig, alpha_orig, n_orig, sum_log_degrees_original = degree_sequence_log_likelihood(original_degree_sequence,
                                                                                           d_min)
    ll_current, alpha_current, n_current, sum_log_degrees_current = degree_sequence_log_likelihood(
        current_degree_sequence, d_min)
    ll_comb, alpha_comb, n_comb, sum_log_degrees_combined = degree_sequence_log_likelihood(concat_degree_sequence,
                                                                                           d_min)
    # Compute the log likelihood ratio
    current_ratio = -2 * ll_comb + 2 * (ll_orig + ll_current)

    # Compute new log likelihood values that would arise if we add/remove the edges corresponding to each node pair.
    new_lls, new_alphas, new_ns, new_sum_log_degrees = updated_log_likelihood_for_edge_changes(node_pairs,
                                                                                               tf.cast(
                                                                                                   modified_adjacency,
                                                                                                   tf.float32), d_min)

    # Combination of the original degree distribution with the distributions corresponding to each node pair.
    n_combined = n_orig + new_ns
    new_sum_log_degrees_combined = sum_log_degrees_original + new_sum_log_degrees
    alpha_combined = compute_alpha(n_combined, new_sum_log_degrees_combined, d_min)
    new_ll_combined = compute_log_likelihood(n_combined, alpha_combined, new_sum_log_degrees_combined, d_min)
    new_ratios = -2 * new_ll_combined + 2 * (new_lls + ll_orig)

    # Allowed edges are only those for which the resulting likelihood ratio measure is < than the threshold
    allowed_edges = new_ratios < threshold
    filtered_edges = tf.boolean_mask(node_pairs, allowed_edges)

    # Get the flattened indices for the allowed edges [e,2] -> [e,], similar to np.ravel_multi_index
    flat_ixs = ravel_multiple_indices(tf.cast(filtered_edges, tf.int32), modified_adjacency.shape)
    # Also for the reverse direction (we assume unweighted graphs).
    flat_ixs_reverse = ravel_multiple_indices(tf.reverse(tf.cast(filtered_edges, tf.int32), [1]),
                                              modified_adjacency.shape)

    # Construct a [N * N] array with ones at the admissible node pair locations and 0 everywhere else.
    indices_1 = tf.scatter_nd(flat_ixs[:, None], tf.ones_like(flat_ixs, dtype=tf.float32), shape=[N * N])
    indices_2 = tf.scatter_nd(flat_ixs_reverse[:, None], tf.ones_like(flat_ixs_reverse, dtype=tf.float32),
                              shape=[N * N])

    # Add both directions
    allowed_mask = tf.clip_by_value(indices_1 + indices_2, 0, 1)

    return allowed_mask, current_ratio
