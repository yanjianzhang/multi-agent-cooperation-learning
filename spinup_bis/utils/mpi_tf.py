"""Distributed Adam Optimizer."""
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from stable_baselines import logger

from spinup_bis.utils import mpi_tools


def flat_concat(xs):
    return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)


def assign_params_from_flat(x, params):
    flat_size = lambda p: int(
        np.prod(p.shape.as_list()))  # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in
                  zip(params, splits)]

    return tf.group([p.assign(p_new) for p, p_new in zip(params, new_params)])


def sync_params(params):
    flat_params = flat_concat(params)

    def _broadcast(x):
        weights = x.numpy()
        mpi_tools.broadcast(weights)
        return weights

    synced_params = tf.py_function(_broadcast, [flat_params], tf.float32)
    return assign_params_from_flat(synced_params, params)

def sync_all_params():
    """Sync all tf variables across MPI processes."""
    return sync_params(tf.compat.v1.global_variables())


class MpiAdamOptimizer(tf.compat.v1.train.AdamOptimizer):
    """Adam optimizer that averages gradients across MPI processes.

    The minimize method is based on compute_gradients taken from Baselines
    `MpiAdamOptimizer`_.

    .. _`MpiAdamOptimizer`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_adam_optimizer.py  # pylint: disable=line-too-long
    """

    def __init__(self, grad_clip=None, mpi_rank_weight=1, **kwargs):
        self._comm = MPI.COMM_WORLD
        self.comm = self._comm
        self.mpi_rank_weight = mpi_rank_weight
        self.grad_clip = grad_clip
        tf.compat.v1.train.AdamOptimizer.__init__(self, **kwargs)

    def minimize(self, loss, var_list, grad_loss=None, name=None):
        """Same as normal minimize, except average grads over processes."""
        grads_and_vars = self.compute_gradients(loss, var_list=var_list,
                                                 grad_loss=grad_loss)

        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self._comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self._comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_function(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                              for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return self.apply_gradients(avg_grads_and_vars, name=name)

    def compute_gradients(self, loss, var_list, **kwargs):
        grads_and_vars = tf.compat.v1.train.AdamOptimizer.compute_gradients(self, loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = tf.concat([tf.reshape(g, (-1,)) for g, v in grads_and_vars], axis=0) * self.mpi_rank_weight
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        total_weight = np.zeros(1, np.float32)
        self.comm.Allreduce(np.array([self.mpi_rank_weight], dtype=np.float32), total_weight, op=MPI.SUM)
        total_weight = total_weight[0]

        buf = np.zeros(sum(sizes), np.float32)
        countholder = [0]  # Counts how many times _collect_grads has been called
        stat = tf.reduce_sum(grads_and_vars[0][1])  # sum of first variable

        def _collect_grads(flat_grad, np_stat):
            if self.grad_clip is not None:
                gradnorm = np.linalg.norm(flat_grad)
                if gradnorm > 1:
                    flat_grad /= gradnorm
                logger.logkv_mean('gradnorm', gradnorm)
                logger.logkv_mean('gradclipfrac', float(gradnorm > 1))
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(total_weight), out=buf)
            if countholder[0] % 100 == 0:
                check_synced(np_stat, self.comm)
            countholder[0] += 1
            return buf

        avg_flat_grad = tf.numpy_function(_collect_grads, [flat_grad, stat], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                              for g, (_, v) in zip(avg_grads, grads_and_vars)]
        return avg_grads_and_vars

def check_synced(localval, comm=None):
    """
    It's common to forget to initialize your variables to the same values, or
    (less commonly) if you update them in some other way than adam, to get them out of sync.
    This function checks that variables on all MPI workers are the same, and raises
    an AssertionError otherwise
    Arguments:
        comm: MPI communicator
        localval: list of local variables (list of variables on current worker to be compared with the other workers)
    """
    comm = comm or MPI.COMM_WORLD
    vals = comm.gather(localval)
    if comm.rank == 0:
        assert all(val==vals[0] for val in vals[1:]),\
            'MpiAdamOptimizer detected that different workers have different weights: {}'.format(vals)