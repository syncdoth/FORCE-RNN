"""
FORCE_INTERNAL_ALL2ALL.py

This function generates the sum of 4 sine waves in figure 2D using the arcitecture of figure 1C (all-to-all
connectivity) with the RLS learning rule.  The all-2-all connectivity allows a large optimization, in that we can
maintain a single inverse correlation matrix for the network.  It's also not as a hard a learning problem as internal
learning with sparse connectivity because there are no approximations of the eigenvectors of the correlation matrices,
as there would be if this was sparse internal learning.  Note that there is no longer a feedback loop from the output
unit.

This is re-written in python from original matlab code.

written by Sehyun Choi
"""
import math
import time

import numpy as np
import matplotlib.pyplot as plt


class HP:
    """Hyperparams."""
    # initialize for plotting
    do_plot = True
    linewidth = 3
    fontsize = 14

    # main params
    N = 1000  # number of neurons  5,000 ~ 20,000
    p = 1.0  # sparsity
    g = 1.3  # 1.2 ~ 1.3
    alpha = 1.0  # learning rate. RLS alpha value.
    learn_every = 1  # learn every step
    nRec2Out = N  # number of neurons that account for output

    # amp lower because wf is implied as all ones, which is half the strength of
    # wf.
    amp = 0.7
    freq = 1 / 60

    nsecs_train = 1500  # number of seconds. Determines iteration of the training.
    nsecs_test = 500  # number of seconds. Determines iteration of the training.
    dt = 0.25
    tau = 2.5

    print('   N: ', N)
    print('   g: ', g)
    print('   p: ', p)
    print('   nRec2Out: ', nRec2Out)
    print('   alpha: ', "{:5.3f}".format(alpha))
    print('   nsecs_train: ', nsecs_train)
    print('   learn_every: ', learn_every)
    print()


class all2all:
    """A network where all neurons are interconnected."""

    def __init__(self, N, p, g, alpha, learn_every, nRec2Out, simtime_len, num_inputs=0, num_outputs=1):
        self.N = N
        self.alpha = alpha
        self.learn_every = learn_every
        self.nRec2Out = nRec2Out

        scale = 1.0 / ((p * N)**(1 / 2))
        self.M = np.random.randn(N, N) * g * scale  # Matrix of all synapses
        self.P = (1.0 / self.alpha) * np.eye(self.nRec2Out)  # [N, N]

        # initial weight matrix for output z calculation.
        self.wo = np.zeros([nRec2Out, num_outputs])  # [N, 1]
        self.dw = np.zeros([nRec2Out, num_outputs])  # [N, 1]


        self.u = np.random.randn(num_inputs, N) if num_inputs > 0 else None

        # initial variables
        self.x0 = 0.5 * np.random.randn(N, 1)  # [N, 1]
        self.z0 = 0.5 * np.random.randn(1)  # [1,]


    def run(self, x, rate, dt, tau, H=None):
        """simulate the network at each timestep.

        essentially, the feedback signal x goes through the neurons, and
        gets modified. This is used to calculate firing rate, r(t).
        """
        dx = (dt / tau) * (-x + self.M @ rate)  # [N, 1]
        if self.u is not None and H is not None:
            dx += (dt / tau) * (self.u.T @ H.T)
        x += dx
        rate = np.tanh(x)  # [N, 1]
        # NOTE: Originally complex conjugate transpose, but we are dealing with
        # real values here, so using .T suffices.
        z = self.wo.T @ rate  # [1,]
        z = z[0]  # to scalar

        return x, rate, z


    def _update(self, rate, z, ft, ti):
        """update inverse correlation matrix and readout weight."""
        # Equation (5) in the paper.
        k = self.P @ rate  # [N, 1]
        rPr = rate.T @ k  # [1, 1]
        c = 1.0 / (1.0 + rPr[0][0])
        self.P = self.P - k @ (k.T * c)

        # update the error for the linear readout
        e = z - ft[ti]

        # update the output weights
        self.dw = -e * k * c  # [N, 1]
        self.wo = self.wo + self.dw  # [N, 1]

        # update the internal weight matrix using the output's error
        self.M = self.M + np.tile(self.dw.T, (self.N, 1))  # [N, N]


    def train(self, simtime, ft, nsecs, dt, tau, H=None, cont=False, cont_var=None,
              num_epochs=1):
        """train"""
        if cont and cont_var is not None:
            x = cont_var['x']
            z = cont_var['z']
        else:
            x = self.x0  # [N, 1]
            z = self.z0  # [1,]
        rate = np.tanh(x)  # [N, 1]
        zt = np.zeros([num_epochs, simtime.shape[0]])  # [N]
        wo_len = np.zeros([num_epochs, simtime.shape[0]])  # [N]
        et = []

        for epoch in range(num_epochs):
            ti = 0
            for t in simtime:
                # TODO: resolve plotting code
                if ti % (nsecs / 2) == 0:
                    print('time: ', "{:.1f}".format(t))

                # sim, so x(t) and r(t) are created.
                Ht = None if H is None else H[ti:ti+1]
                x, rate, z = self.run(x, rate, dt, tau, Ht)

                if ti % self.learn_every == 0:
                    self._update(rate, z, ft, ti)

                # Store the output of the system.
                zt[epoch, ti] = z
                wo_len[epoch, ti] = np.sqrt(self.wo.T @ self.wo)
                ti += 1

            error_avg = np.sum(np.abs(zt[epoch] - ft)) / simtime.shape[0]
            et.append(error_avg)
            print('Training MAE for epoch {}: "{:5.3f}"'.format(epoch, error_avg))

        if HP.do_plot:
            # TODO: Make this iterative plotting, following original code.
            fig, axs = plt.subplots(3, figsize=(10, 10))
            fig.suptitle('training', fontsize=HP.fontsize)
            axs[0].set_xlabel('time', fontsize=HP.fontsize)
            axs[0].set_ylabel('f and z', fontsize=HP.fontsize)
            line1, = axs[0].plot(
                simtime, ft, linewidth=HP.linewidth, color='green')
            line2, = axs[0].plot(
                simtime, zt[-1], linewidth=HP.linewidth, color='red')
            axs[0].legend((line1, line2), ('ft', 'zt'), loc='upper right')

            axs[1].set_xlabel('time', fontsize=HP.fontsize)
            axs[1].set_ylabel('|w|', fontsize=HP.fontsize)
            axs[1].plot(simtime, wo_len[-1], linewidth=HP.linewidth)

            axs[2].set_xlabel('epochs', fontsize=HP.fontsize)
            axs[2].set_ylabel('MAE', fontsize=HP.fontsize)
            axs[2].plot(np.arange(0, num_epochs, 1), et, linewidth=HP.linewidth)
        return x, rate, zt


    def test(self, simtime_test, ft_test, x, rate, dt, tau, H=None, num_epochs=1):
        """test"""
        print('Now testing... please wait.')
        zt_predict = np.zeros([num_epochs, simtime_test.shape[0]])  # [N]
        et = []
        for epoch in range(num_epochs):
            ti = 0
            for _ in simtime_test:  # don't want to subtract time in indices
                Ht = None if H is None else H[ti:ti+1]
                x, rate, z = self.run(x, rate, dt, tau, Ht)
                zt_predict[epoch, ti] = z
                ti += 1

            error_avg = np.sum(
                np.abs(zt_predict[epoch] - ft_test)) / simtime_test.shape[0]
            et.append(error_avg)
        print('Testing MAE: ', "{:5.3f}".format(error_avg))
        if HP.do_plot:
            plt.figure()
            plt.title('simulation', fontsize=HP.fontsize)
            plt.xlabel('time', fontsize=HP.fontsize)
            plt.ylabel('f and z', fontsize=HP.fontsize)
            line3, = plt.plot(
                simtime_test, ft_test, linewidth=HP.linewidth, color='green')
            line4, = plt.plot(
                simtime_test, zt_predict[-1], linewidth=HP.linewidth, color='red')
            plt.legend((line3, line4), ('ft_test','zt_predict'), loc='upper right')
        return zt_predict


def get_ft(simtime, amp, freq):
    """get default ft setting. 4 sin waves."""
    ft = ((amp / 1.0) * np.sin(1.0 * math.pi * freq * simtime) +
          (amp / 2.0) * np.sin(2.0 * math.pi * freq * simtime) +
          (amp / 6.0) * np.sin(3.0 * math.pi * freq * simtime) +
          (amp / 3.0) * np.sin(4.0 * math.pi * freq * simtime))
    return ft / 1.5
