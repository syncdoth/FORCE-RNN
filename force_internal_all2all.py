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
    fontweight = 'bold'

    # main params
    N = 1000  # number of neurons
    p = 1.0  # ???
    g = 1.5  # g greater than 1 leads to chaotic networks.
    alpha = 1.0  # learning rate. RLS alpha value.
    learn_every = 2  # iterations to update weight.
    nRec2Out = N  # number of neurons that account for output

    amp = 0.7  # amp lower because wf is implied as all ones, which is half the strength of wf.
    freq = 1 / 60

    nsecs = 1440  # number of seconds. Determines iteration of the training.
    dt = 0.1  # time step value.

    print('   N: ', N)
    print('   g: ', g)
    print('   p: ', p)
    print('   nRec2Out: ', nRec2Out)
    print('   alpha: ', "{:5.3f}".format(alpha))
    print('   nsecs: ', nsecs)
    print('   learn_every: ', learn_every)
    print()


class all2all:
    def __init__(self, N, p, g, alpha, learn_every, nRec2Out, simtime_len):
        self.N = N
        self.alpha = alpha
        self.learn_every = learn_every
        self.nRec2Out = nRec2Out

        scale = 1.0 / ((p * N)**(1 / 2))
        self.M = np.random.randn(N, N) * g * scale  # Matrix of all synapses
        self.P = (1.0 / self.alpha) * np.eye(self.nRec2Out)  # [N, N]

        # initial weight matrix for output z calculation.
        self.wo = np.zeros([nRec2Out, 1])  # [N, 1]
        self.dw = np.zeros([nRec2Out, 1])  # [N, 1]

        # initial variables
        self.x0 = 0.5 * np.random.randn(N, 1)  # [N, 1]
        self.z0 = 0.5 * np.random.randn(1)  # [1,]


    def _simulate(self, x, rate, dt):
        x = (1.0 - dt) * x + self.M @ (rate * dt)  # [N, 1]
        rate = np.tanh(x)  # [N, 1]
        # NOTE: Originally complex conjugate transpose, but we are dealing with
        # real values here, so using .T suffices.
        z = self.wo.T @ rate  # [1,]
        z = z[0]  # to scalar

        return x, rate, z


    def _update(self, rate, z, ft, ti):
        # update inverse correlation matrix
        k = self.P @ rate  # [N, 1]
        rPr = rate.T @ k  # [1, 1]
        c = 1.0 / (1.0 + rPr[0][0])
        self.P = self.P - k @ (k.T * c)  # [N, N] - [N, N] = [N, N]

        # update the error for the linear readout
        e = z - ft[ti]

        # update the output weights
        self.dw = -e * k * c  # [N, 1]
        self.wo = self.wo + self.dw  # [N, 1]

        # update the internal weight matrix using the output's error
        self.M = self.M + np.tile(self.dw.T, (self.N, 1))  # [N, N]


    def train(self, simtime, ft, nsecs, dt):
        """train"""
        x = self.x0  # [N, 1]
        rate = np.tanh(x)  # [N, 1]
        z = self.z0  # [1,]
        zt = np.zeros([simtime.shape[0]])  # [N]
        wo_len = np.zeros([simtime.shape[0]])  # [N]

        ti = 0
        for t in simtime:
            # TODO: resolve plotting code
            if ti % (nsecs / 2) == 0:
                print('time: ', "{:.1f}".format(t))

            # sim, so x(t) and r(t) are created.
            x, rate, z = self._simulate(x, rate, dt)

            if ti % self.learn_every == 0:
                self._update(rate, z, ft, ti)

            # Store the output of the system.
            zt[ti] = z
            wo_len[ti] = np.sqrt(self.wo.T @ self.wo)
            ti += 1

        error_avg = np.sum(np.abs(zt - ft)) / simtime.shape[0]
        print('Training MAE: ', "{:5.3f}".format(error_avg))

        if HP.do_plot:
            # TODO: Make this iterative plotting, following original code.
            fig, axs = plt.subplots(2, figsize=(10, 10))
            fig.suptitle('training', fontsize=HP.fontsize)
            axs[0].set_xlabel('time', fontsize=HP.fontsize)
            axs[0].set_ylabel('f and z', fontsize=HP.fontsize)
            line1, = axs[0].plot(
                simtime, ft, linewidth=HP.linewidth, color='green')
            line2, = axs[0].plot(
                simtime, zt, linewidth=HP.linewidth, color='red')
            axs[0].legend((line1, line2), ('ft', 'zt'), loc='upper right')

            axs[1].set_xlabel('time', fontsize=HP.fontsize)
            axs[1].set_ylabel('|w|', fontsize=HP.fontsize)
            axs[1].plot(simtime, wo_len, linewidth=HP.linewidth)
        return x, rate, zt


    def test(self, simtime_test, ft_test, x, rate, dt):
        """test"""
        print('Now testing... please wait.')
        zt_predict = np.zeros([simtime_test.shape[0]])  # [N]
        ti = 0
        for _ in simtime_test:  # don't want to subtract time in indices
            x, rate, z = self._simulate(x, rate, dt)
            zt_predict[ti] = z
            ti += 1

        error_avg = np.sum(
            np.abs(zt_predict - ft_test)) / simtime_test.shape[0]
        print('Testing MAE: ', "{:5.3f}".format(error_avg))
        return zt_predict


def get_ft(simtime, amp, freq):
    """get default ft setting. 4 sin waves."""
    ft = ((amp / 1.0) * np.sin(1.0 * math.pi * freq * simtime) +
          (amp / 2.0) * np.sin(2.0 * math.pi * freq * simtime) +
          (amp / 6.0) * np.sin(3.0 * math.pi * freq * simtime) +
          (amp / 3.0) * np.sin(4.0 * math.pi * freq * simtime))
    return ft / 1.5


def main():
    # simulation time array
    simtime = np.arange(0, HP.nsecs - HP.dt, HP.dt)
    simtime_test = np.arange(1 * HP.nsecs, 2 * HP.nsecs - HP.dt, HP.dt)

    # target function ft
    ft = get_ft(simtime, HP.amp, HP.freq)
    ft_test = get_ft(simtime_test, HP.amp, HP.freq)

    network = all2all(HP.N, HP.p, HP.g, HP.alpha, HP.learn_every, HP.nRec2Out,
                      simtime.shape[0])

    x, rate, zt = network.train(simtime, ft, HP.nsecs, HP.dt)
    zt_predict = network.test(simtime_test, ft_test, x, rate, HP.dt)

    if HP.do_plot:
        fig, axs = plt.subplots(2, figsize=(10, 10))
        fig
        line1, = axs[0].plot(simtime, ft, linewidth=HP.linewidth, color='green')
        axs[0].set_title('training', fontsize=HP.fontsize)
        axs[0].set_xlabel('time', fontsize=HP.fontsize)
        axs[0].set_ylabel('f and z', fontsize=HP.fontsize)
        line2, = axs[0].plot(simtime, zt, linewidth=HP.linewidth, color='red')
        axs[0].legend((line1, line2), ('ft', 'zt'), loc='upper right')

        axs[1].set_title('simulation', fontsize=HP.fontsize)
        axs[1].set_xlabel('time', fontsize=HP.fontsize)
        axs[1].set_ylabel('f and z', fontsize=HP.fontsize)
        line3, = axs[1].plot(
            simtime_test, ft_test, linewidth=HP.linewidth, color='green')
        line4, = axs[1].plot(
            simtime_test, zt_predict, linewidth=HP.linewidth, color='red')
        axs[1].legend((line3, line4), ('ft_test', 'zt_predict'),
                      loc='upper right')
        plt.show()


if __name__ == "__main__":
    main()
