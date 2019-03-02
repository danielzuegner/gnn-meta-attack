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
from metattack import utils
import scipy.sparse as sp
from tensorflow.contrib import slim

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, desc=None: x


class GNNAttack:
    """
        Base class for attacks on GNNs.
    """
    def __init__(self, adjacency_matrix, attribute_matrix, labels_onehot, hidden_sizes, train_iters=100, gpu_id=None,
                 attack_features=False, dtype=tf.float32):
        """

        Parameters
        ----------
        adjacency_matrix: np.array [N,N]
                Unweighted, symmetric adjacency matrix where N is the number of nodes.

        attribute_matrix: sp.spmatrix or np.array [N,D]
            Attribute matrix where D is the number of attributes per node.

        labels_onehot: np.array [N,K]
            One-hot matrix of class labels, where N is the number of nodes. Labels of the unlabeled nodes should come
            from self-training using only the labels of the labeled nodes.

        hidden_sizes: list of ints
            List that defines the number of hidden units per hidden layer. Input and output layers not included.

        train_iters: int
            The number of 'inner' training steps of the GCN

        gpu_id: int or None
            GPU to use. None means CPU-only

        attack_features: bool
            Whether to also attack the node attributes (in addition to the graph structure).

        """

        self.N, self.D = attribute_matrix.shape
        self.K = labels_onehot.shape[1]
        self.hidden_sizes = hidden_sizes
        self.graph = tf.Graph()
        self.train_iters = train_iters

        self.dtype = dtype

        with self.graph.as_default():

            self.labels_onehot = labels_onehot
            self.idx_labeled = tf.placeholder(dtype=tf.int32, shape=[None, ], name="Labeled_Idx")
            self.idx_unlabeled = tf.placeholder(dtype=tf.int32, shape=[None, ], name="Unlabeled_Idx")
            self.idx_attack = tf.placeholder(dtype=tf.int32, shape=[None, ], name="Attack_Idx")
            self.attack_features = attack_features

            if sp.issparse(adjacency_matrix):
                adjacency_matrix = adjacency_matrix.toarray()

            assert np.allclose(adjacency_matrix, adjacency_matrix.T)
            self.sparse_attributes = sp.issparse(attribute_matrix)
            if attack_features:
                if self.sparse_attributes:
                    attrs_unique = np.unique(attribute_matrix.toarray())
                    # convert attributes to dense to make them attackable
                    attribute_matrix = attribute_matrix.toarray()
                    self.sparse_attributes = False
                else:
                    attrs_unique = np.unique(attribute_matrix)
                if len(attrs_unique) > 2 or not np.allclose(attrs_unique, [0, 1]):
                    raise ValueError("Attacks on the node features are currently only supported for binary attributes.")

            w_init = slim.xavier_initializer
            weights = []
            biases = []
            velocities = []
            bias_velocities = []

            previous_size = self.D
            for ix, layer_size in enumerate(self.hidden_sizes):
                weight = tf.get_variable(f"W_{ix + 1}", shape=[previous_size, layer_size], dtype=self.dtype,
                                         initializer=w_init())
                bias = tf.get_variable(f"b_{ix + 1}", shape=[layer_size], dtype=self.dtype,
                                       initializer=w_init())
                w_velocity = tf.Variable(np.zeros(weight.shape), dtype=self.dtype, name=f"Velocity_{ix + 1}")
                b_velocity = tf.Variable(np.zeros(bias.shape), dtype=self.dtype, name=f"b_Velocity_{ix + 1}")
                weights.append(weight)
                velocities.append(w_velocity)
                bias_velocities.append(b_velocity)
                biases.append(bias)
                previous_size = layer_size

            output_weight = tf.get_variable(f"W_{len(self.hidden_sizes) + 1}", shape=[previous_size, self.K],
                                            dtype=self.dtype,
                                            initializer=w_init())
            output_bias = tf.get_variable(f"b_{len(self.hidden_sizes) + 1}", shape=[self.K], dtype=self.dtype,
                                          initializer=w_init())
            output_velocity = tf.Variable(np.zeros(output_weight.shape), dtype=self.dtype,
                                          name=f"Velocity_{len(self.hidden_sizes) + 1}")
            output_bias_velocity = tf.Variable(np.zeros(output_bias.shape), dtype=self.dtype,
                                               name=f"b_Velocity_{len(self.hidden_sizes) + 1}")
            weights.append(output_weight)
            velocities.append(output_velocity)

            biases.append(output_bias)
            bias_velocities.append(output_bias_velocity)

            with tf.name_scope("input"):
                self.adjacency_orig = tf.constant(adjacency_matrix, dtype=self.dtype, name="Adjacency")

                # The variable storing the changes to the adjacency matrix. Shape [N*N]
                self.adjacency_changes = tf.Variable(np.zeros(adjacency_matrix.size), dtype=self.dtype,
                                                     name="Adjacency_delta")

                # reshape to [N, N] and set the diagonal to 0
                tf_adjacency_square = tf.matrix_set_diag(tf.reshape(self.adjacency_changes, adjacency_matrix.shape),
                                                         tf.zeros(adjacency_matrix.shape[0], dtype=self.dtype))

                # Symmetrize and clip to [-1,1]
                tf_adjacency_delta_symm = tf.clip_by_value(tf_adjacency_square + tf.transpose(tf_adjacency_square), -1,
                                                           1)

                self.modified_adjacency = self.adjacency_orig + tf_adjacency_delta_symm

                adj_selfloops = tf.add(self.modified_adjacency, tf.diag(tf.ones([self.N], dtype=self.dtype)))
                inv_degrees = tf.pow(tf.reduce_sum(adj_selfloops, axis=0), -0.5)
                self.adj_norm = tf.multiply(tf.multiply(adj_selfloops, inv_degrees[:, None]),
                                            inv_degrees[None, :], name="normalized_adjacency")

                if attack_features:

                    self.attributes_orig = tf.constant(attribute_matrix, name="Original_attributes",
                                                       dtype=self.dtype)
                    self.attribute_changes = tf.Variable(np.zeros(attribute_matrix.size), dtype=self.dtype)
                    tf_attributes_reshaped = tf.reshape(tf.clip_by_value(self.attribute_changes, 0, 1),
                                                        attribute_matrix.shape)
                    self.attributes = tf.clip_by_value(self.attributes_orig + tf_attributes_reshaped, 0, 1,
                                                       name="Modified_attributes")
                else:
                    if self.sparse_attributes:
                        self.attributes = tf.SparseTensor(np.array(attribute_matrix.nonzero()).T,
                                                          attribute_matrix[attribute_matrix.nonzero()].A1,
                                                          attribute_matrix.shape)
                        self.attributes = tf.cast(self.attributes, dtype=dtype, name="Attributes_sparse")
                    else:
                        self.attributes = tf.constant(attribute_matrix, name="Attribute_matrix", dtype=self.dtype)

            self.all_weights = [[w for w in weights]]
            self.all_biases = [[b for b in biases]]
            self.all_velocities = [[w for w in velocities]]
            self.all_velocities_bias = [[w for w in bias_velocities]]

            if gpu_id is None:
                config = tf.ConfigProto(
                    device_count={'GPU': 0}
                )
            else:
                gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
                config = tf.ConfigProto(gpu_options=gpu_options)

            session = tf.Session(config=config)
            self.session = session

    def filter_potential_singletons(self):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.

        Returns
        -------
        tf.Tensor shape [N, N], float with ones everywhere except the entries of potential singleton nodes,
        where the returned tensor has value 0.

        """
        degrees = tf.reduce_sum(self.modified_adjacency, axis=0)
        degree_one = tf.equal(degrees, 1, name="degree_equals_one")
        resh = tf.reshape(tf.tile(degree_one, [self.N]), [self.N, self.N], name="degree_one_square")
        l_and = tf.logical_and(resh, tf.equal(self.modified_adjacency, 1))
        logical_and_symmetric = tf.logical_or(l_and, tf.transpose(l_and))
        flat_mask = tf.cast(tf.logical_not(tf.reshape(logical_and_symmetric, [-1])), self.dtype)
        return flat_mask

    def log_likelihood_constraint(self, ll_cutoff):
        """
        Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
        log likelihood constraint to be violated.

        Parameters
        ----------
        ll_cutoff: float
            Cutoff value for the unnoticeability constraint. Smaller means stricter constraint. 0.004 corresponds to a
            p-value of 0.95 in the Chi-square distribution with one degree of freedom.

        Returns
        -------
        allowed_mask: tf.Tensor shape [N, N], dtype float
            ones everywhere except the entries that, if an edge is added/removed, would violate the log likelihood
            constraint. There, the returned tensor has value 0.

        current_ratio: tf.Tensor, scalar, dtype float
            current value of the Chi-square test.
        """

        t_d_min = tf.constant(2, dtype=self.dtype)
        t_possible_edges = tf.constant(np.array(np.triu(np.ones((self.N, self.N)), k=1).nonzero()).T,
                                       dtype=tf.uint16)
        allowed_mask, current_ratio = utils.likelihood_ratio_filter(t_possible_edges,
                                                                    self.modified_adjacency,
                                                                    self.adjacency_orig, t_d_min,
                                                                    ll_cutoff)

        return allowed_mask, current_ratio


class GNNMetaApprox(GNNAttack):
    """
    Class for attacking GNNs with approximate meta gradients.
    """
    def __init__(self, adjacency_matrix, attribute_matrix, labels_onehot, hidden_sizes, train_iters=100, gpu_id=None,
                 _lambda=0.5, dtype=tf.float32):
        """

        Parameters
        ----------
        adjacency_matrix: np.array [N,N]
                Unweighted, symmetric adjacency matrix where N is the number of nodes.

        attribute_matrix: sp.spmatrix or np.array [N,D]
            Attribute matrix where D is the number of attributes per node.

        labels_onehot: np.array [N,K]
            One-hot matrix of class labels, where N is the number of nodes. Labels of the unlabeled nodes should come
            from self-training using only the labels of the labeled nodes.

        hidden_sizes: list of ints
            List that defines the number of hidden units per hidden layer. Input and output layers not included.

        train_iters: int
            The number of 'inner' training steps of the GCN

        gpu_id: int or None
            GPU to use. None means CPU-only

        _lambda: float between 0 and 1 (inclusive)
            Weighting of the gradients of the losses of the labeled and unlabeled nodes. _lambda=1 corresponds to only
            considering the loss on the labeled nodes, _lambda=0 only unlabeled nodes.

        """

        super().__init__(adjacency_matrix, attribute_matrix, labels_onehot, hidden_sizes, train_iters, gpu_id,
                         False, dtype)

        self.lambda_ = _lambda
        self.logits = None
        self.classification_loss = None
        self.optimizer = None
        self.train_op = None
        self.grad_sum = None
        self.adjacency_grad = None
        self.grad_sum_add = None
        self.grad_sum_mod = None
        self.adjacency_update = None
        self.ll_ratio = None

    def build(self, with_relu=False, learning_rate=1e-2):
        """
        Construct the model and create the weight variables.
        Parameters
        ----------
        with_relu: bool
            Whether to use the ReLU activation in the hidden layers
        learning_rate: float
            Learning rate for training.

        """
        with self.graph.as_default():

            weights = self.all_weights[-1]
            bias = self.all_biases[-1]

            hidden = self.attributes
            for ix, w in enumerate(weights):
                b = bias[ix]
                if ix == 0 and self.sparse_attributes:
                    if self.dtype != tf.float32:  # sparse matmul is unfortunately not implemented for float16
                        hidden = self.adj_norm @ tf.cast(tf.sparse_tensor_dense_matmul(tf.cast(hidden, tf.float32),
                                                                                       tf.cast(w, tf.float32)),
                                                         self.dtype) + b
                    else:
                        hidden = self.adj_norm @ tf.sparse_tensor_dense_matmul(hidden, w) + b
                else:
                    hidden = self.adj_norm @ hidden @ w + b
                if with_relu:
                    hidden = tf.nn.relu(hidden)

            self.logits = hidden

            labels_gather = tf.gather(self.labels_onehot, self.idx_labeled)
            logits_gather = tf.gather(self.logits, self.idx_labeled)
            self.classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_gather,
                                                                                                 logits=logits_gather))
            epsilon = 1e-8
            if self.dtype == tf.float16:
                epsilon = 1e-4  # improve numerical stability for half precision
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
            self.train_op = self.optimizer.minimize(self.classification_loss, var_list=[*self.all_weights[0],
                                                                                        *self.all_biases[0]])

    def make_loss(self, ll_constraint=True, ll_cutoff=0.004):
        """
        Construct the update of the adjacency matrix based on the (approximate) meta gradients.

        Parameters
        ----------
        ll_constraint: bool
            Whether to enforce the unnoticeability constraint on the degree distribution.

        ll_cutoff: float
            Cutoff value for the unnoticeability constraint. Smaller means stricter constraint. 0.004 corresponds to a
            p-value of 0.95 in the Chi-square distribution with one degree of freedom.

        """

        with self.graph.as_default():

            logits_labeled = tf.gather(self.logits, self.idx_labeled)
            labels_train = tf.gather(self.labels_onehot, self.idx_labeled)
            logits_unlabeled = tf.gather(self.logits, self.idx_unlabeled)
            labels_selftrain = tf.gather(self.labels_onehot, self.idx_unlabeled)

            loss_labeled = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_labeled,
                                                                                     labels=labels_train))
            loss_unlabeled = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_unlabeled,
                                                                                       labels=labels_selftrain))

            if self.lambda_ == 1:
                attack_loss = loss_labeled
            elif self.lambda_ == 0:
                attack_loss = loss_unlabeled
            else:
                attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

            # This variable "stores" the gradients of every inner training step.
            self.grad_sum = tf.Variable(np.zeros(self.N * self.N), dtype=self.dtype)

            self.adjacency_grad = tf.multiply(tf.gradients(attack_loss, self.adjacency_changes)[0],
                                              tf.reshape(self.modified_adjacency, [-1]) * -2 + 1,
                                              name="Adj_gradient")
            # Add the current gradient to the sum.
            self.grad_sum_add = tf.assign_add(self.grad_sum, self.adjacency_grad)

            # Make sure that the minimum entry is 0.
            self.grad_sum_mod = self.grad_sum - tf.reduce_min(self.grad_sum)

            # Set entries to 0 that could lead to singleton nodes.
            singleton_mask = self.filter_potential_singletons()
            self.grad_sum_mod = tf.multiply(self.grad_sum_mod, singleton_mask)

            if ll_constraint:
                print("Enforcing likelihood ratio constraint with cutoff {}".format(ll_cutoff))
                allowed_mask, self.ll_ratio = self.log_likelihood_constraint(ll_cutoff)

                # Set entries to 0 that would violate the log likelihood constraint.
                self.grad_sum_mod = tf.multiply(self.grad_sum_mod, allowed_mask)

            # Get argmax of the approximate meta gradients.
            adj_meta_approx_argmax = tf.argmax(self.grad_sum_mod)

            # Compute the index corresponding to the reverse direction of the edge (i.e. in the other triangle of the
            # matrix).
            adj_argmax_transpose_ix = utils.ravel_index(utils.unravel_index_tf(adj_meta_approx_argmax,
                                                                               [self.N, self.N])[::-1],
                                                        [self.N, self.N])
            # Stack both indices to make sure our matrix remains symmetric.
            adj_argmax_combined = tf.stack([adj_meta_approx_argmax, adj_argmax_transpose_ix],
                                           name="Meta_approx_argmax_combined")

            # Add the change to the perturbations.
            self.adjacency_update = tf.scatter_add(self.adjacency_changes,
                                                   indices=adj_argmax_combined,
                                                   updates=-2 * tf.gather(
                                                            tf.reshape(self.modified_adjacency, [-1]),
                                                            adj_argmax_combined) + 1)

    def attack(self, perturbations, idx_labeled, idx_unlabeled, idx_attack, initialize=True):
        """
        Perform the attack on the surrogate model.

        Parameters
        ----------
        perturbations: int
            The number of changes to perform.

        idx_labeled: np.array of ints
            The indices of the labeled nodes.

        idx_unlabeled: np.array of ints
            The indices of the unlabeled nodes.

        idx_attack: np.array of ints
            The indices of the nodes to be attacked (e.g. the indices of the unlabeled nodes).

        initialize: bool, default: True
            Whether to initialize all variables before beginning.

        """
        with self.graph.as_default():
            if initialize:
                self.session.run(tf.global_variables_initializer())

            weights = [w for v in self.all_weights for w in v]
            biases = [b for v in self.all_biases for b in v]
            opt_vars = [v for v in self.optimizer.variables()]

            for _it in tqdm(range(perturbations), desc="Perturbing graph"):
                self.session.run(self.grad_sum.initializer)
                self.session.run([v.initializer for v in [*weights, *biases, *opt_vars]])
                for tr_iter in range(self.train_iters):
                    self.session.run([self.train_op, self.grad_sum_add],
                                     {self.idx_labeled: idx_labeled, self.idx_unlabeled: idx_unlabeled})
                self.session.run(self.adjacency_update,
                                 {self.idx_attack: idx_attack, self.idx_labeled: idx_labeled,
                                  self.idx_unlabeled: idx_unlabeled})


class GNNMeta(GNNAttack):
    """
    Class for attacking GNNs with meta gradients.
    """

    def __init__(self, adjacency_matrix, attribute_matrix, labels_onehot, hidden_sizes, train_iters=100, gpu_id=None,
                 attack_features=False, dtype=tf.float32):
        """

        Parameters
        ----------
        adjacency_matrix: np.array [N,N]
                Unweighted, symmetric adjacency matrix where N is the number of nodes.

        attribute_matrix: sp.spmatrix or np.array [N,D]
            Attribute matrix where D is the number of attributes per node.

        labels_onehot: np.array [N,K]
            One-hot matrix of class labels, where N is the number of nodes. Labels of the unlabeled nodes should come
            from self-training using only the labels of the labeled nodes.

        hidden_sizes: list of ints
            List that defines the number of hidden units per hidden layer. Input and output layers not included.

        train_iters: int
            The number of 'inner' training steps of the GCN

        gpu_id: int or None
            GPU to use. None means CPU-only

        attack_features: bool
            Whether to also attack the node attributes (in addition to the graph structure).

        """

        super().__init__(adjacency_matrix, attribute_matrix, labels_onehot, hidden_sizes, train_iters, gpu_id,
                         attack_features, dtype)

        self.logits_final = None
        self.adjacency_meta_grad = None
        self.adjacency_meta_update = None
        self.attribute_meta_grad = None
        self.attribute_meta_update = None
        self.combined_update = None
        self.ll_ratio = None

    def build(self, with_relu=False, learning_rate=0.1, momentum=0.9):
        """
        Construct the model and create the weight variables.

        Parameters
        ----------
        with_relu: bool
            Whether to use the ReLU activation in the hidden layers

        learning_rate: float
            Learning rate for training.

        momentum: float
            Momentum term for SGD with momentum.

        """

        with self.graph.as_default():
            with tf.name_scope("training"):
                # Unroll the training and store all the weights explicitly.
                for it in tqdm(range(self.train_iters), desc="Unrolling training procedure"):

                    current_weights = self.all_weights[-1]
                    current_biases = self.all_biases[-1]
                    current_velocities_bias = self.all_velocities_bias[-1]
                    current_velocities = self.all_velocities[-1]

                    hidden = self.attributes
                    for ix, w in enumerate(current_weights):
                        b = current_biases[ix]
                        if ix == 0 and self.sparse_attributes:
                            hidden = self.adj_norm @ tf.sparse_tensor_dense_matmul(hidden, w) + b
                        else:
                            hidden = self.adj_norm @ hidden @ w + b
                        if with_relu:
                            hidden = tf.nn.relu(hidden)

                    logits_train = tf.gather(hidden, self.idx_labeled)
                    loss_per_node = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_train,
                                                                               labels=tf.gather(self.labels_onehot,
                                                                                                self.idx_labeled))
                    loss = tf.reduce_mean(loss_per_node)

                    weight_grads = tf.gradients(loss, current_weights)
                    bias_grads = tf.gradients(loss, current_biases)
                    next_velocities = [momentum * current_v + weight_grads[ix] for ix, current_v in
                                       enumerate(current_velocities)]
                    next_b_velocities = [momentum * v + bias_grads[ix] for ix, v in enumerate(current_velocities_bias)]

                    # Perform updates on the weights and biases and add them to the lists of weights/biases.
                    next_weights = [tf.subtract(w, learning_rate * next_velocities[ix]) for ix, w in
                                    enumerate(current_weights)]
                    next_biases = [tf.subtract(b, learning_rate * next_b_velocities[ix]) for ix, b in
                                   enumerate(current_biases)]

                    self.all_weights.append(next_weights)
                    self.all_biases.append(next_biases)
                    self.all_velocities.append(next_velocities)
                    self.all_velocities_bias.append(next_b_velocities)

            final_weights = self.all_weights[-1]
            final_bias = self.all_biases[-1]

            final_output = self.attributes
            for ix, w in enumerate(final_weights):
                b = final_bias[ix]
                if ix == 0 and self.sparse_attributes:
                    final_output = self.adj_norm @ tf.sparse_tensor_dense_matmul(final_output, w) + b
                else:
                    final_output = self.adj_norm @ final_output @ w + b
                if with_relu:
                    final_output = tf.nn.relu(final_output)

            self.logits_final = final_output

    def make_loss(self, ll_constraint=True, ll_cutoff=0.004):
        """
        Construct the update of the adjacency matrix based on the meta gradients.

        Parameters
        ----------
        ll_constraint: bool
            Whether to enforce the unnoticeability constraint on the degree distribution.

        ll_cutoff: float
            Cutoff value for the unnoticeability constraint. Smaller means stricter constraint. 0.004 corresponds to a
            p-value of 0.95 in the Chi-square distribution with one degree of freedom.

        """
        with self.graph.as_default():

            logits_attack = tf.gather(self.logits_final, self.idx_attack)
            labels_atk = tf.gather(self.labels_onehot, self.idx_attack)
            attack_loss_per_node = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_attack,
                                                                              labels=labels_atk)
            attack_loss = tf.reduce_mean(attack_loss_per_node)

            # Meta gradient computation.
            self.adjacency_meta_grad = tf.multiply(tf.gradients(attack_loss, self.adjacency_changes)[0],
                                                   tf.reshape(self.modified_adjacency, [-1]) * -2 + 1,
                                                   name="Meta_gradient")

            # Make sure that the minimum entry is 0.
            self.adjacency_meta_grad -= tf.reduce_min(self.adjacency_meta_grad)

            # Set entries to 0 that could lead to singleton nodes.
            singleton_mask = self.filter_potential_singletons()
            self.adjacency_meta_grad = tf.multiply(self.adjacency_meta_grad, singleton_mask)

            if ll_constraint:
                print("Enforcing likelihood ratio constraint with cutoff {}".format(ll_cutoff))
                allowed_mask, self.ll_ratio = self.log_likelihood_constraint(ll_cutoff)

                # Set entries to 0 that would violate the log likelihood constraint.
                self.adjacency_meta_grad = tf.multiply(self.adjacency_meta_grad, allowed_mask)

            # Get argmax of the meta gradients.
            adj_meta_grad_argmax = tf.argmax(self.adjacency_meta_grad)

            # Compute the index corresponding to the reverse direction of the edge (i.e. in the other triangle of the
            # matrix).
            adj_argmax_transpose_ix = utils.ravel_index(utils.unravel_index_tf(adj_meta_grad_argmax,
                                                                               [self.N, self.N])[::-1],
                                                        [self.N, self.N])
            # Stack both indices to make sure our matrix remains symmetric.
            adj_argmax_combined = tf.stack([adj_meta_grad_argmax, adj_argmax_transpose_ix],
                                           name="Meta_grad_argmax_combined")

            # Add the change to the perturbations.
            self.adjacency_meta_update = tf.scatter_add(self.adjacency_changes,
                                                        indices=adj_argmax_combined,
                                                        updates=-2 * tf.gather(
                                                            tf.reshape(self.modified_adjacency, [-1]),
                                                            adj_argmax_combined) + 1)

            if self.attack_features:
                # Get meta gradients of the attributes.
                self.attribute_meta_grad = tf.multiply(tf.gradients(attack_loss, self.attribute_changes)[0],
                                                       tf.reshape(self.attributes, [-1]) * -2 + 1)
                self.attribute_meta_grad -= tf.reduce_min(self.attribute_meta_grad)

                attribute_meta_grad_argmax = tf.argmax(self.attribute_meta_grad)

                self.attribute_meta_update = tf.scatter_add(self.attribute_changes,
                                                            indices=attribute_meta_grad_argmax,
                                                            updates=-2 * tf.gather(
                                                                tf.reshape(self.attributes, [-1]),
                                                                attribute_meta_grad_argmax) + 1),

                adjacency_meta_grad_max = tf.reduce_max(self.adjacency_meta_grad)
                attribute_meta_grad_max = tf.reduce_max(self.attribute_meta_grad)

                # If the meta gradient of the structure meta gradient is larger, we perform a structure perturbation.
                # Otherwise, we change an attribute.
                cond = adjacency_meta_grad_max > attribute_meta_grad_max

                self.combined_update = tf.cond(cond, lambda: self.adjacency_meta_update,
                                               lambda: self.attribute_meta_update)

    def attack(self, perturbations, idx_labeled, idx_attack, initialize=True):
        """
        Perform the attack on the surrogate model.

        Parameters
        ----------
        perturbations: int
            The number of changes to perform.

        idx_labeled: np.array of ints
            The indices of the labeled nodes.

        idx_attack: np.array of ints
            The indices of the nodes to be attacked (e.g. the indices of the unlabeled nodes).

        initialize: bool, default: True
            Whether to initialize all variables before beginning.

        """

        with self.graph.as_default():
            if initialize:
                self.session.run(tf.global_variables_initializer())

            for _it in tqdm(range(perturbations), desc="Perturbing graph"):
                self.session.run(self.adjacency_meta_update,
                                 {self.idx_attack: idx_attack, self.idx_labeled: idx_labeled})


class GCNSparse:
    """
    GCN implementation with a sparse adjacency matrix and possibly sparse attribute matrices. Note that this becomes
    the surrogate model from the paper if we set the number of layers to 2 and leave out the ReLU activation function
    (see build()).
    """

    def __init__(self, adjacency_matrix, attribute_matrix, labels_onehot, hidden_sizes, gpu_id=None):
        """
        Parameters
        ----------
        adjacency_matrix: sp.spmatrix [N,N]
                Unweighted, symmetric adjacency matrix where N is the number of nodes. Should be a scipy.sparse matrix.

        attribute_matrix: sp.spmatrix or np.array [N,D]
            Attribute matrix where D is the number of attributes per node. Can be sparse or dense.

        labels_onehot: np.array [N,K]
            One-hot matrix of class labels, where N is the number of nodes. Labels of the unlabeled nodes should come
            from self-training using only the labels of the labeled nodes.

        hidden_sizes: list of ints
            List that defines the number of hidden units per hidden layer. Input and output layers not included.

        gpu_id: int or None
            GPU to use. None means CPU-only

        """
        if not sp.issparse(adjacency_matrix):
            raise ValueError("Adjacency matrix should be a sparse matrix.")

        self.N, self.D = attribute_matrix.shape
        self.K = labels_onehot.shape[1]
        self.hidden_sizes = hidden_sizes
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.idx = tf.placeholder(tf.int32, shape=[None])
            self.labels_onehot = labels_onehot

            adj_norm = utils.preprocess_graph(adjacency_matrix).astype("float32")
            self.adj_norm = tf.SparseTensor(np.array(adj_norm.nonzero()).T,
                                            adj_norm[adj_norm.nonzero()].A1, [self.N, self.N])

            self.sparse_attributes = sp.issparse(attribute_matrix)

            if self.sparse_attributes:
                self.attributes = tf.SparseTensor(np.array(attribute_matrix.nonzero()).T,
                                                  attribute_matrix[attribute_matrix.nonzero()].A1, [self.N, self.D])
            else:
                self.attributes = tf.Variable(attribute_matrix, dtype=tf.float32)

            w_init = slim.xavier_initializer
            self.weights = []
            self.biases = []

            previous_size = self.D
            for ix, layer_size in enumerate(self.hidden_sizes):
                weight = tf.get_variable(f"W_{ix + 1}", shape=[previous_size, layer_size], dtype=tf.float32,
                                         initializer=w_init())
                bias = tf.get_variable(f"b_{ix + 1}", shape=[layer_size], dtype=tf.float32,
                                       initializer=w_init())
                self.weights.append(weight)
                self.biases.append(bias)
                previous_size = layer_size
            weight_final = tf.get_variable(f"W_{len(hidden_sizes) + 1}", shape=[previous_size, self.K],
                                           dtype=tf.float32,
                                           initializer=w_init())
            bias_final = tf.get_variable(f"b_{len(hidden_sizes) + 1}", shape=[self.K], dtype=tf.float32,
                                         initializer=w_init())

            self.weights.append(weight_final)
            self.biases.append(bias_final)

            if gpu_id is None:
                config = tf.ConfigProto(
                    device_count={'GPU': 0}
                )
            else:
                gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
                config = tf.ConfigProto(gpu_options=gpu_options)

            session = tf.Session(config=config)
            self.session = session

            self.logits = None
            self.logits_gather = None
            self.loss = None
            self.optimizer = None
            self.train_op = None
            self.initializer = None

    def build(self, with_relu=True, learning_rate=1e-2):
        with self.graph.as_default():
            hidden = self.attributes
            for ix in range(len(self.hidden_sizes)):
                w = self.weights[ix]
                b = self.biases[ix]
                if ix == 0 and self.sparse_attributes:
                    hidden = tf.sparse_tensor_dense_matmul(self.adj_norm,
                                                           tf.sparse_tensor_dense_matmul(self.attributes, w)) + b
                else:
                    hidden = tf.sparse_tensor_dense_matmul(self.adj_norm, self.attributes @ w) + b

                if with_relu:
                    hidden = tf.nn.relu(hidden)

            self.logits = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden @ self.weights[-1]) + self.biases[-1]
            self.logits_gather = tf.gather(self.logits, self.idx)
            labels_gather = tf.gather(self.labels_onehot, self.idx)
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_gather, logits=self.logits_gather)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, var_list=[*self.weights, *self.biases])
            self.initializer = tf.local_variables_initializer()

    def train(self, idx_train, n_iters=200, initialize=True, display=True):
        with self.graph.as_default():
            if initialize:
                self.session.run(tf.global_variables_initializer())

            _iter = range(n_iters)
            if display:
                _iter = tqdm(_iter, desc="Training")

            for _it in _iter:
                self.session.run(self.train_op, feed_dict={self.idx: idx_train})
