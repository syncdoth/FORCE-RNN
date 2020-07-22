"""
This is implemented in reference to sample code from FORCE Paper's supplementary
data.

written by Sehyun Choi
"""
import math
import time
import random

import numpy as np
import matplotlib.pyplot as plt


class HP:
    """Hyperparams."""
    # initialize for plotting
    do_plot = True
    linewidth = 3
    fontsize = 14

    # main params
    N = 400  # number of neurons
    g = 1.5  # g greater than 1 leads to chaotic networks.
    alpha = 1.0  # learning rate. RLS alpha value.
    learn_every = 2  # iterations to update weight.

    # amp lower because wf is implied as all ones, which is half the strength of
    # wf.
    amp = 0.7
    freq = 1 / 60

    nsecs = 100  # number of seconds. Determines iteration of the training.
    dt = 0.01  # time step value.

    print('   N: ', N)
    print('   g: ', g)
    print('   alpha: ', "{:5.3f}".format(alpha))
    print('   nsecs: ', nsecs)
    print('   learn_every: ', learn_every)
    print()

class Network:

    # TODO: using HP.N directly might cause problem later on.
    u_in = np.random.uniform(low=-1.0, high=1.0, size=[HP.N, 1])

    def __init__(self, N, learn_every):
        self.N = N
        self.learn_every = learn_every
        self.H = np.tanh

    def _update_P(self, P, rate):
        """update inverse correlation matrix."""
        k = P @ rate  # [N, 1]
        rPr = rate.T @ k  # [1, 1]
        c = 1.0 / (1.0 + rPr[0][0])
        P = P - k @ (rate.T @ P) * c
        return P

    def _activity(self, rate, dt):
        raise NotImplementedError

    def _update(self, rate, z, ft, ti):
        raise NotImplementedError


class TaskPerforming(Network):
    def __init__(self, N, learn_every, g, alpha):
        super().__init__(N, learn_every)

        # initial weight matrix for output z calculation.
        self.J = np.zeros((N, N))  # Matrix of all synapses
        self.P_J = (1.0 / alpha) * np.eye(N)  # [N, N]
        self.P_w = (1.0 / alpha) * np.eye(N)  # [N, N]
        self.w = np.random.randn(N, 1)  # [N, 1]

        # initial variables
        self.x = 0.5 * np.random.randn(N, 1)  # [N, 1]
        self.z = 0.5 * np.random.randn(1)  # [1,]


    def _activity(self, dt, ext_input=None):
        dx = -(dt * self.x) + self.J @ (self.H(self.x) * dt)
        if ext_input:
            dx += self.u_in * (ext_input * dt)
        self.x += dx
        z = self.w.T @ self.H(self.x)  # [1,]
        self.z = z[0]  # to scalar

    def _update_w(self, target):
        rate = self.H(self.x)
        self.P_w = self._update_P(self.P_w, rate)
        # update the error for the linear readout
        e = self.z - target
        # update the output weights
        dw = -e * self.P_w @ rate  # [N, 1]
        self.w += dw  # [N, 1]

    def _update_J(self, J_D, x_D, u, target):
        rate = self.H(self.x)

        self.P_J = self._update_P(self.P_J, rate)
        e = self.J @ rate - J_D @ self.H(x_D) - u * target
        # update the output weights
        dJ = -e.T @ self.P_J @ rate  # [1, 1]
        self.J += dJ

        return e

    def get_delta_t(self):
        delta_t = int(random.gauss(mu=2.0, sigma=1.0))
        while delta_t == 0:
            delta_t = int(random.gauss(mu=2.0, sigma=1.0))
        return delta_t

    def train_J(self, simtime, f_out, nsecs, dt, target_net, f_in=None):
        """train"""
        et = np.zeros([simtime.shape[0]])
        zdt = np.zeros([simtime.shape[0]])
        zt = np.zeros([simtime.shape[0]])

        ti = 0
        for t in simtime:
            ext_input = 0
            target = f_out[ti]
            if ti % (nsecs / 2) == 0:
                print('time: ', "{:.1f}".format(t))

            # sim, so x(t) and r(t) are created.
            self._activity(dt, ext_input)
            target_net._activity(dt, ext_input, target)

            learn_every = self.get_delta_t()
            if ti % learn_every == 0:
                e = self._update_J(target_net.J_D, target_net.x_D, target_net.u, target)
                learn_every = self.get_delta_t()

            # Store the output of the system.
            et[ti] = np.sum(e) / self.N
            zdt[ti] = self.w.T @ self.H(target_net.x_D)
            zt[ti] = self.z
            ti += 1

        error_avg = np.sum(et) / simtime.shape[0]
        print('Training MAE for J: ', "{:5.3f}".format(error_avg))

        if HP.do_plot:
            # TODO: Make this iterative plotting, following original code.
            fig, axs = plt.subplots(2, figsize=(8, 8))
            fig.suptitle('training', fontsize=HP.fontsize)
            axs[0].set_xlabel('time', fontsize=HP.fontsize)
            axs[0].set_ylabel('z_d and z', fontsize=HP.fontsize)
            line1, = axs[0].plot(
                simtime, zdt, linewidth=HP.linewidth, color='green')
            line2, = axs[0].plot(
                simtime, zt, linewidth=HP.linewidth, color='red')
            axs[0].legend((line1, line2), ('zdt', 'zt'), loc='upper right')

            axs[1].set_xlabel('time', fontsize=HP.fontsize)
            axs[1].set_ylabel('|e|', fontsize=HP.fontsize)
            axs[1].plot(simtime, et, linewidth=HP.linewidth)


    def train_w(self, simtime, f_out, nsecs, dt, target_net, f_in=None):
        """train"""
        zt = np.zeros([simtime.shape[0]])  # [N]
        wo_len = np.zeros([simtime.shape[0]])  # [N]

        ti = 0
        for t in simtime:
            ext_input = f_in[ti] if f_in is not None else None
            target = f_out[ti]
            if ti % (nsecs / 2) == 0:
                print('time: ', "{:.1f}".format(t))

            # sim, so x(t) and r(t) are created.
            self._activity(dt, ext_input)

            if ti % self.learn_every == 0:
                self._update_w(target)

            # Store the output of the system.
            zt[ti] = self.z
            wo_len[ti] = np.sqrt(self.w.T @ self.w)
            ti += 1

        error_avg = np.sum(np.abs(zt - f_out)) / simtime.shape[0]
        print('Training MAE: ', "{:5.3f}".format(error_avg))

        if HP.do_plot:
            # TODO: Make this iterative plotting, following original code.
            fig, axs = plt.subplots(2, figsize=(8, 8))
            fig.suptitle('training', fontsize=HP.fontsize)
            axs[0].set_xlabel('time', fontsize=HP.fontsize)
            axs[0].set_ylabel('f and z', fontsize=HP.fontsize)
            line1, = axs[0].plot(
                simtime, f_out, linewidth=HP.linewidth, color='green')
            line2, = axs[0].plot(
                simtime, zt, linewidth=HP.linewidth, color='red')
            axs[0].legend((line1, line2), ('ft', 'zt'), loc='upper right')

            axs[1].set_xlabel('time', fontsize=HP.fontsize)
            axs[1].set_ylabel('|w|', fontsize=HP.fontsize)
            axs[1].plot(simtime, wo_len, linewidth=HP.linewidth)
        return zt


    def test(self, simtime_test, f_out_test, dt, f_in_test=None):
        """test"""
        print('Now testing... please wait.')
        zt_predict = np.zeros([simtime_test.shape[0]])  # [N]
        ti = 0
        for _ in simtime_test:  # don't want to subtract time in indices
            ext_input = f_in_test[ti] if f_in_test is not None else None
            self._activity(dt, ext_input)
            zt_predict[ti] = self.z
            ti += 1

        error_avg = np.sum(
            np.abs(zt_predict - f_out_test)) / simtime_test.shape[0]
        print('Testing MAE: ', "{:5.3f}".format(error_avg))
        return zt_predict


class TargetGenerating(Network):
    def __init__(self, N, learn_every, g, alpha):
        super().__init__(N, learn_every)

        # initial variables
        self.J_D = np.random.normal(scale=g / (N ** (1 / 2)), size=(N, N))  # Matrix of all synapses
        self.x_D = 0.5 * np.random.randn(N, 1)  # [N, 1]
        self.u = np.random.uniform(low=-1.0, high=1.0, size=[N, 1])

        # for testing
        self.w_D = np.zeros([N, 1])
        self.P = np.eye(N) / alpha
        self.z_D = 0.5 * np.random.randn(1)  # [1,]


    def _activity(self, dt, ext_input, target):
        dx = -(dt * self.x_D) + self.J_D @ (self.H(self.x_D) * dt) + self.u * target * dt
        if ext_input:
            dx += self.u_in * (ext_input * dt)
        self.x_D += dx

        z = self.w_D.T @ self.H(self.x_D)  # [1,]
        self.z_D = z[0]  # to scalar

    def _update_w(self, target):
        # update the error for the linear readout
        e = self.z_D - target
        # update the output weights
        dw = -e * self.P @ self.H(self.x_D)  # [N, 1]
        # print(dw)
        self.w_D += dw  # [N, 1]

    def _update(self, target):
        self.P = self._update_P(self.P, self.H(self.x_D))
        self._update_w(target)

    def test(self, simtime, f_out, nsecs, dt, f_in=None):
        """train"""
        zt = np.zeros([simtime.shape[0]])  # [nsecs]
        wo_len = np.zeros([simtime.shape[0]])  # [nsecs]
        Jt = np.zeros([simtime.shape[0]])

        ti = 0
        for t in simtime:
            ext_input = f_in[ti] if f_in is not None else None
            target = f_out[ti]
            if ti % (nsecs / 2) == 0:
                print('time: ', "{:.1f}".format(t))

            # sim, so x(t) and r(t) are created.
            self._activity(dt, ext_input, target)

            if ti % self.learn_every == 0:
                self._update(target)

            # Store the output of the system.
            zt[ti] = self.z_D
            wo_len[ti] = np.sqrt(self.w_D.T @ self.w_D)
            Jt[ti] = self.J_D[0][0]
            ti += 1

        error_avg = np.sum(np.abs(zt - f_out)) / nsecs
        print('Training MAE: ', "{:5.3f}".format(error_avg))
        if HP.do_plot:
            # TODO: Make this iterative plotting, following original code.
            fig, axs = plt.subplots(3, figsize=(8, 8))
            fig.suptitle('training', fontsize=HP.fontsize)
            axs[0].set_xlabel('time', fontsize=HP.fontsize)
            axs[0].set_ylabel('f and z', fontsize=HP.fontsize)
            line1, = axs[0].plot(
                simtime[-500:], f_out[-500:], linewidth=HP.linewidth, color='green')
            line2, = axs[0].plot(
                simtime[-500:], zt[-500:], linewidth=HP.linewidth, color='red')
            axs[0].legend((line1, line2), ('ft', 'zt'), loc='upper right')

            axs[1].set_xlabel('time', fontsize=HP.fontsize)
            axs[1].set_ylabel('|w|', fontsize=HP.fontsize)
            axs[1].plot(simtime[-500:], wo_len[-500:], linewidth=HP.linewidth)
            axs[2].set_xlabel('time', fontsize=HP.fontsize)
            axs[2].set_ylabel('f_in', fontsize=HP.fontsize)
            axs[2].plot(simtime[-500:], f_in[-500:], linewidth=HP.linewidth)
            plt.show()

        return zt


def osciliation_task(dt):
    step = math.pi * (6 - 2) * dt
    wt = np.arange(math.pi * 2, math.pi * 6, step)
    half = np.arange(0, 1, dt)
    period = np.sin(wt * half)
    period = np.concatenate([period, -period[::-1]])

    ft = np.tile(period, 50)
    pulse = np.concatenate([np.ones([10]), np.zeros([190])])
    f_in = np.tile(pulse, 50)
    return ft, f_in


def main():
    # simulation time array
    simtime = np.arange(0, HP.nsecs, HP.dt)
    simtime_test = np.arange(HP.nsecs, 2 * HP.nsecs, HP.dt)

    # target function ft
    f_out, f_in = osciliation_task(HP.dt)
    f_out_test, f_in_test = osciliation_task(HP.dt)

    task_net = TaskPerforming(HP.N, HP.learn_every, HP.g, HP.alpha)
    target_net = TargetGenerating(HP.N, HP.learn_every, HP.g, HP.alpha)

    # target_net.test(simtime, f_out, HP.nsecs, HP.dt, f_in=f_in)

    task_net.train_J(simtime, f_out, HP.nsecs, HP.dt, target_net, f_in=f_in)
    zt = task_net.train_w(simtime, f_out, HP.nsecs, HP.dt, target_net, f_in=f_in)
    zt_predict = task_net.test(simtime_test, f_out_test, HP.dt, f_in_test=f_in_test)

    if HP.do_plot:
        fig, axs = plt.subplots(2, figsize=(8, 8))
        line1, = axs[0].plot(simtime, f_out, linewidth=HP.linewidth, color='green')
        axs[0].set_title('training', fontsize=HP.fontsize)
        axs[0].set_xlabel('time', fontsize=HP.fontsize)
        axs[0].set_ylabel('f and z', fontsize=HP.fontsize)
        line2, = axs[0].plot(simtime, zt, linewidth=HP.linewidth, color='red')
        axs[0].legend((line1, line2), ('ft', 'zt'), loc='upper right')

        axs[1].set_title('simulation', fontsize=HP.fontsize)
        axs[1].set_xlabel('time', fontsize=HP.fontsize)
        axs[1].set_ylabel('f and z', fontsize=HP.fontsize)
        line3, = axs[1].plot(
            simtime_test, f_out_test, linewidth=HP.linewidth, color='green')
        line4, = axs[1].plot(
            simtime_test, zt_predict, linewidth=HP.linewidth, color='red')
        axs[1].legend((line3, line4), ('ft_test', 'zt_predict'),
                      loc='upper right')
        plt.show()


if __name__ == "__main__":
    main()
