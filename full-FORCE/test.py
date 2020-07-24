import numpy as np
import matplotlib.pyplot as plt
import time

def fullforce_oscillation_test(dt, showplots=0):
    dt_per_s = round(1/dt)

    # From the paper, and the online demo:
    t = np.expand_dims(np.linspace(0,2,2*dt_per_s+1),1)
    omega = np.zeros((2*dt_per_s+1,1))
    omega = np.linspace(2*np.pi, 6*np.pi, 1*dt_per_s+1)
    targ = np.zeros((2*dt_per_s+1,1))
    targ[0:(1*dt_per_s+1),0] = np.sin(t[0:(1*dt_per_s+1),0]*omega)
    targ[1*dt_per_s:(2*dt_per_s+1)] = -np.flipud(targ[0:(1*dt_per_s+1)])

    # A simpler example: just a sine wave
    '''
    t = np.expand_dims(np.linspace(0,2,2*dt_per_s+1),1)
    omega = np.ones((2*dt_per_s+1,1)) * 4 *np.pi
    targ = np.sin(t*omega)
    '''

    # A slightly harder example: sum of sine waves
    '''
    t = np.expand_dims(np.linspace(0,2,2*dt_per_s+1),1)
    omega = np.ones((2*dt_per_s+1,1)) * 4 *np.pi
    targ = np.sin(t*2*omega) * np.sin(t*omega/4)
    '''

    inp = np.zeros(targ.shape)
    inp[0:round(0.05*dt_per_s),0] = np.ones((round(0.05*dt_per_s)))
    hints = np.zeros(targ.shape)

    if showplots == 1:
        plt.figure()
        plt.plot(targ)
        plt.plot(hints)
        plt.plot(inp)
        plt.legend(['Target','Hints','Input'])
        plt.show()

    return inp, targ, hints

import Full_FORCE_Sehyun as FF_Demo

p = FF_Demo.create_parameters(dt=0.001)
p['g'] = 1.5 # From paper
p['ff_num_batches'] = 10
p['ff_trials_per_batch'] = 10
p['test_init_trials']=5

rnn = FF_Demo.RNN(p,1,1)

rnn.train(fullforce_oscillation_test, monitor_training=1)

rnn.test(fullforce_oscillation_test)