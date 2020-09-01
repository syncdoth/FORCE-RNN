"""
FF_Demo: A module file for creating, training, and testing recurrent neural
networks using full-FORCE. Created by Eli Pollock, Jazayeri Lab, MIT 12/13/2017

Rewritten by Sehyun Choi
"""

import numpy as np
import numpy.random as npr
from scipy import sparse
import matplotlib.pyplot as plt


def create_parameters(dt=1, network_size=300):
    """Use this to define hyperparameters for any RNN instantiation. You can
    create an "override" script to edit any individual parameters, but simply
    calling this function suffices. Use the output dictionary to instantiate an
    RNN class.
    """
    p = {
        'network_size': network_size,  # Number of units in network
        'dt': dt,  # Time step for RNN.
        'tau': 1,  # Time constant of neurons
        'noise_std': 0,  # Amount of noise on RNN neurons
        'g': 1,  # Gain of the network (scaling for recurrent weights)
        'p': 1,  # Controls the sparseness of the network. 1=>fully connected.
        'inp_scale': 1,  # Scales the initialization of input weights
        'out_scale': 1,  # Scales the initialization of output weights
        'bias_scale': 0,  # Scales the amount of bias on each unit
        'init_act_scale': 1,  # Scales how activity is initialized
        ##### Training parameters for full-FORCE
        'num_epochs': 100,
        'ff_steps_per_update': 1,
        'ff_alpha':
        1,  # "Learning rate" parameter (should be between 1 and 100)
        'ff_init_trials': 3,
        #### Testing parameters
        'test_init_trials': 1,
    }
    return p


class RNN:
    """
    Creates an RNN object. Relevant methods:
        __init___: 			Creates attributes for hyperparameters, network
                            parameters, and initial activity
        initialize_act: 	Resets the activity
        run: 				Runs the network forward given some input
        train:				Uses one of several algorithms to train the network.
    """
    def sigmoid(self, x):
            return 1/(1 + np.exp(-x))


    def __init__(self, hyperparameters, num_inputs, num_outputs):
        """Initialize the network
        Inputs:
            hyperparameters: should be output of create_parameters function,
            changed as needed
            num_inputs: number of inputs into the network
            num_outputs: number of outputs the network has
        Outputs:
            self.hp: Assigns hyperparameters to an attribute
            self.input_weights: weight for inputs; the inputs consist of
            f_in, f_out, and f_hint. [num_inputs, N]
            self.J: strength of synapses. [N, N]
            self.w: readout unit weight. [N, 1]
            self.bias: bias to the activity. [1, N]
            self.x: Initializes the activity of the network [1, N]
        """
        self.hp = hyperparameters
        N = self.hp['network_size']
        self.input_weights = (npr.rand(num_inputs, N) -
                              0.5) * 2 * self.hp['inp_scale']
        self.J = np.array(
            sparse.random(
                N, N, density=self.hp['p'],
                data_rvs=npr.randn).todense()) * self.hp['g'] / np.sqrt(N)
        self.w = (npr.rand(N, num_outputs) - 0.5) * 2 * self.hp['out_scale']
        self.bias = (npr.rand(1, N) - 0.5) * 2 * self.hp['bias_scale']
        self.x = npr.randn(1, N) * self.hp['init_act_scale']
        self.activations = {
            'sigmoid': self.sigmoid,
            'tanh': np.tanh,
        }

    def initialize_act(self):
        """Any time you want to reset the activity of the network to low random
        values.
        """
        self.x = npr.randn(1, self.J.shape[0]) * self.hp['init_act_scale']


    def run(self, inputs, record_flag=0, recurrent_activation='tanh',
            output_activation=None):
        """Use this method to run the RNN on some inputs
        Inputs:
            inputs: An Nxm array, where m is the number of separate inputs and N
            is the number of time steps
            record_flag: If set to 0, the function only records the output node
            activity. If set to 1, if records that and the activity of every
            hidden node over time.
        Outputs:
            output_whole: An Nxk array, where k is the number of separate
            outputs and N is the number of time steps.
            activity_whole: Either an empty array if record_flag=0, or an
            NxQ array, where Q is the number of hidden units.

        """
        hp = self.hp
        x = self.x

        def rnn_update(inp, x):
            dx = hp['dt'] / hp['tau'] * (
                -x + self.activations[recurrent_activation](x) @ self.J +
                inp @ self.input_weights +
                self.bias + npr.randn(1, x.shape[1]) * hp['noise_std'])
            return x + dx

        def rnn_output(x):
            z = self.activations[recurrent_activation](x) @ self.w
            if output_activation:
                z = self.activations[output_activation](z)
            return z


        activity_whole = []
        output_whole = []
        if record_flag == 1:  # Record output and activity only if this is active
            activity_whole = np.zeros(((inputs.shape[0]), self.J.shape[0]))
            t = 0

        for inp in inputs:
            x = rnn_update(inp, x)
            output_whole.append(rnn_output(x))
            if record_flag == 1:
                activity_whole[t, :] = x
                t += 1
        output_whole = np.reshape(output_whole, (inputs.shape[0], -1))
        self.x = x

        return output_whole, activity_whole

    def train(self, inps_and_targs, DRNN=None, monitor_training=0, **kwargs):
        """Use this method to train the RNN using one of several training algorithms!
        Inputs:
            inps_and_targs: This should be a FUNCTION that randomly produces a training input and a target function
                            Those should have individual inputs/targets as columns and be the first two outputs
                            of this function. Should also take a 'dt' argument.
            monitor_training: Collect useful statistics and show at the end
            **kwargs: use to pass things to the inps_and_targs function
        Outputs:
        Nothing explicitly, but the weights of self.rnn_par are optimized to map the inputs to the targets
        Use this to train the network according to the full-FORCE algorithm, described in DePasquale 2017
        This function uses a recursive least-squares algorithm to optimize the network.
        Note that after each batch, the function shows an example output as well as recurrent unit activity.
        Parameters:
            In self.hp, the parameters starting with ff_ control this function.

        *****NOTE***** The function inps_and_targs must have a third output of "hints" for training.
        If you don't want to use hints, replace with a vector of zeros (Nx1)

        """
        # First, initialize some parameters
        hp = self.hp
        self.initialize_act()
        N = hp['network_size']
        # Initialize to zero for training.
        self.J = np.zeros((N, N))
        self.w = np.zeros((self.w.shape))

        # Need to initialize a target-generating network, used for computing error:
        # First, take some example inputs, targets, and hints to get the right shape
        try:
            inputs, targs, hints = inps_and_targs(dt=hp['dt'],
                                                  **kwargs)[0:3]
            num_inputs = inputs[0].shape[1]
            num_targs = targs[0].shape[1]
            num_hints = hints[0].shape[1]
            num_total_inps = num_inputs + num_targs + num_hints
        except:
            raise ValueError(
                'Check your inps_and_targs function. Must have a hints output as well!'
            )

        # Then instantiate the network and pull out some relevant weights
        if DRNN is None:
            DRNN = RNN(hyperparameters=self.hp,
                       num_inputs=num_total_inps,
                       num_outputs=self.w.shape[1])
            Jd = np.transpose(DRNN.J)  # [N, N]
            w_targ = np.transpose(  # [N, num_outputs]
            DRNN.input_weights[num_inputs:(num_inputs + num_targs), :])
            w_hint = np.transpose(  # [N, num_outputs]
                DRNN.input_weights[(num_inputs +
                                    num_targs):num_total_inps, :])
        else:
            Jd = DRNN.M
            w_targ = np.transpose(  # [N, num_outputs]
                DRNN.u[num_inputs:(num_inputs + num_targs), :])
            w_hint = np.transpose(  # [N, num_outputs]
                DRNN.u[(num_inputs + num_targs):num_total_inps, :])

        ################### Monitor training with these variables:
        J_err_ratio = []
        J_err_mag = []
        J_norm = []

        w_err_ratio = []
        w_err_mag = []
        w_norm = []

        error_history = []
        ###################

        # Let the networks settle from the initial conditions
        print('Initializing', end="")
        for i in range(hp['ff_init_trials']):
            print('.', end="")
            D_total_inp = np.hstack((inputs[0], targs[0], hints[0]))
            if isinstance(DRNN, RNN):
                DRNN.run(D_total_inp)
            else:
                x_D = npr.randn(N, 1) * self.hp['init_act_scale']
                x_D = DRNN.run(x_D, np.tanh(x_D), hp['dt'], hp['tau'], D_total_inp[i:i+1])[0]
            self.run(inputs[0])
        print('')

        # Now begin training
        print('Training network...')
        # Monitor whole activity
        z_whole = []
        dx_whole = []
        x_whole = []
        targ_whole = []
        # Initialize the inverse correlation matrix
        P = np.eye(N) / hp['ff_alpha']

        if isinstance(inputs, np.ndarray):
            input_len = inputs.shape[0]
        elif isinstance(inputs, list):
            input_len = len(inputs)
        else:
            raise TypeError("input must be a list or numpy array!")
        for epoch in range(hp['num_epochs']):
            print(f'epoch {epoch+1} of {hp["num_epochs"]}, ',
                  f'{input_len} examples: ')
            inp, targ, hint = inps_and_targs(dt=hp['dt'],
                                             **kwargs)[0:3]

            epoch_error = []
            for trial in range(input_len):
                if np.mod(trial, int(input_len / 10)) == 0:
                    # print(f'progress.. {trial}/{input_len}')
                    print('.', end='')

                batch = {
                    'inp': inp[trial],
                    'targ': targ[trial],
                    'hint': hint[trial],
                }
                D_total_inp = np.hstack(
                    (batch['inp'], batch['targ'], batch['hint']))
                # For recording:
                dx = []  # Driven network activity
                x = []  # RNN activity
                z = []  # RNN output
                for t in range(batch['inp'].shape[0]):
                    if t % (batch['inp'].shape[0] / 10) == 0:
                        print('.', end='')
                    # Run both RNNs forward and get the activity. Record activity for potential plotting
                    if isinstance(DRNN, RNN):
                        dx_t = DRNN.run(D_total_inp[t:t + 1, :],
                                        record_flag=1)[1][:, 0:2]
                        x_D = DRNN.x
                    else:
                        x_D = DRNN.run(x_D, np.tanh(x_D), hp['dt'], hp['tau'], D_total_inp[t:t + 1, :])[0]
                        dx_t = np.expand_dims(x_D, axis=0)
                    z_t, x_t = self.run(batch['inp'][t:t + 1, :], record_flag=1)

                    dx.append(np.squeeze(np.tanh(dx_t) + np.arange(2) * 2))
                    z.append(np.squeeze(z_t))
                    x.append(
                        np.squeeze(np.tanh(x_t[:, 0:2]) + np.arange(2) * 2))

                    if t % hp['ff_steps_per_update'] == 0:
                        # Extract relevant values
                        r = self.activations['tanh'](self.x).T  # [N, 1]
                        rd = self.activations['tanh'](x_D) # [N, 1]
                        rd = rd.T if isinstance(DRNN, RNN) else rd
                        J = self.J.T           # [N, N]
                        w = self.w.T           # [1, N]

                        # Now for the RLS algorithm:
                        # Compute errors
                        J_err = (J @ r - Jd @ rd -  # [N, 1]
                                 (w_targ @ batch['targ'][t:t + 1, :].T) -
                                 (w_hint @ batch['hint'][t:t + 1, :].T))
                        w_err = w @ r - batch['targ'][t:t + 1, :].T

                        # Compute the gain (k) and running estimate of the inverse correlation matrix
                        Pr = P @ r
                        # Originally, k is not transposed. I think this is done
                        # to match the dimension when calculating the update
                        # terms.
                        k = Pr.T / (1 + r.T @ Pr)
                        P = P - Pr @ k

                        # Update weights
                        w = w - w_err @ k  # [1, 1] @ [1, 300] = [1, 300]
                        J = J - J_err @ k  # [300, 1] @ [1, 300] = [300, 300]
                        self.J = J.T
                        self.w = w.T

                        if monitor_training == 1:
                            J_err_plus = (
                                J @ r - Jd @ rd -
                                w_targ @ batch['targ'][t:t + 1, :].T -
                                w_hint @ batch['hint'][t:t + 1, :].T)
                            J_err_ratio = np.hstack(
                                (J_err_ratio,
                                 np.squeeze(np.mean(J_err_plus / J_err))))
                            J_err_mag = np.hstack(
                                (J_err_mag, np.squeeze(np.linalg.norm(J_err))))
                            J_norm = np.hstack(
                                (J_norm, np.squeeze(np.linalg.norm(J))))

                            w_err_plus = w @ r - batch['targ'][t:t + 1, :].T
                            w_err_ratio = np.hstack(
                                (w_err_ratio, np.squeeze(w_err_plus / w_err)))
                            w_err_mag = np.hstack(
                                (w_err_mag, np.squeeze(np.linalg.norm(w_err))))
                            w_norm = np.hstack(
                                (w_norm, np.squeeze(np.linalg.norm(w))))
                            epoch_error.append(np.sum(np.abs(w_err)) / w_err.shape[1])
                z_whole.append(z)
                dx_whole.append(dx)
                x_whole.append(x)
                targ_whole.append(targ)
            ########## Epoch callback
            print('mean epoch error: ', sum(epoch_error) / (input_len * inp[0].shape[0]))
            print('')  # New line after each epoch

            # Convert lists to arrays
            dx = np.array(dx)
            x = np.array(x)
            z = np.array(z)

            # if epoch == 0:
            #     # Set up plots
            #     training_fig = plt.figure()
            #     ax_unit = training_fig.add_subplot(2, 1, 1)
            #     ax_out = training_fig.add_subplot(2, 1, 2)
            #     tvec = np.arange(0, len(inp)) * hp['dt']

            #     # Create output and target lines
            #     lines_targ_out = plt.Line2D(tvec,
            #                                 targ,
            #                                 linestyle='--',
            #                                 color='r')
            #     lines_out = plt.Line2D(tvec, z, color='b')
            #     ax_out.add_line(lines_targ_out)
            #     ax_out.add_line(lines_out)

            #     # Create recurrent unit and DRNN target lines
            #     lines_targ_unit = {}
            #     lines_unit = {}
            #     for i in range(2):
            #         lines_targ_unit['%g' % i] = plt.Line2D(tvec,
            #                                                dx[:, i],
            #                                                linestyle='--',
            #                                                color='r')
            #         lines_unit['%g' % i] = plt.Line2D(tvec, x[:, i], color='b')
            #         ax_unit.add_line(lines_targ_unit['%g' % i])
            #         ax_unit.add_line(lines_unit['%g' % i])

            #     # Set up the axes
            #     ax_out.set_xlim([0, hp['dt'] * len(inp)])
            #     ax_unit.set_xlim([0, hp['dt'] * len(inp)])
            #     ax_out.set_ylim([-1.2, 1.2])
            #     ax_unit.set_ylim([-2, 10])
            #     ax_out.set_title('Output')
            #     ax_unit.set_title('Recurrent units, batch %g' % (epoch + 1))

            #     # Labels
            #     ax_out.set_xlabel('Time (s)')
            #     ax_out.legend([lines_targ_out, lines_out], ['Target', 'RNN'],
            #                   loc=1)
            # else:
            #     # Update the plot
            #     tvec = np.arange(0, len(inp)) * hp['dt']
            #     ax_out.set_xlim([0, hp['dt'] * len(inp)])
            #     ax_unit.set_xlim([0, hp['dt'] * len(inp)])
            #     ax_unit.set_title('Recurrent units, batch %g' % (epoch + 1))
            #     lines_targ_out.set_xdata(tvec)
            #     lines_targ_out.set_ydata(targ)
            #     lines_out.set_xdata(tvec)
            #     lines_out.set_ydata(z)
            #     for i in range(2):
            #         lines_targ_unit['%g' % i].set_xdata(tvec)
            #         lines_targ_unit['%g' % i].set_ydata(dx[:, i])
            #         lines_unit['%g' % i].set_xdata(tvec)
            #         lines_unit['%g' % i].set_ydata(x[:, i])
            # training_fig.canvas.draw()

        if monitor_training == 1:
            # Now for some visualization to see how things went:
            stats_fig = plt.figure(figsize=(8, 10))
            plt.subplot(4, 2, 1)
            plt.title('Recurrent learning error ratio')
            plt.plot(J_err_ratio)

            plt.subplot(4, 2, 3)
            plt.title('Recurrent error magnitude')
            plt.plot(J_err_mag)

            plt.subplot(4, 2, 5)
            plt.title('Recurrent weights norm')
            plt.plot(J_norm)

            plt.subplot(4, 2, 2)
            plt.plot(w_err_ratio)
            plt.title('Output learning error ratio')

            plt.subplot(4, 2, 4)
            plt.plot(w_err_mag)
            plt.title('Output error magnitude')

            plt.subplot(4, 2, 6)
            plt.plot(w_norm)
            plt.title('Output weights norm')

            plt.subplot(4, 2, 7)
            plt.plot(epoch_error)
            plt.title('error history')

            plt.subplot(4, 2, 8)
            plt.plot(np.squeeze(np.array(targ_whole[-1])))
            plt.plot(np.squeeze(np.array(z_whole[-1])))
            plt.title('output')
            stats_fig.canvas.draw()
        print('Done training!')
        return z_whole

    def test(self, inps_and_targs, **kwargs):
        hp = self.hp
        """
        Function that tests a trained network. Relevant parameters in p start with 'test'
        Inputs:
            Inps_and_targ: function used to generate time series (same as in train)
            **kwargs: arguments passed to inps_and_targs
        """

        self.initialize_act()
        print('Initializing', end="")

        inp, targ = inps_and_targs(dt=hp['dt'], **kwargs)[0:2]
        for _ in range(hp['test_init_trials']):
            print('.', end="")
            self.run(inp[0])
        print('')

        test_fig = plt.figure()
        ax = test_fig.add_subplot(1, 1, 1)
        tvec = np.arange(0, inp.shape[1]) * hp['dt']
        line_inp = plt.Line2D(tvec, targ, linestyle='--', color='g')
        line_targ = plt.Line2D(tvec, targ, linestyle='--', color='r')
        line_out = plt.Line2D(tvec, targ, color='b')
        ax.add_line(line_inp)
        ax.add_line(line_targ)
        ax.add_line(line_out)
        ax.legend([line_inp, line_targ, line_out],
                  ['Input', 'Target', 'Output'],
                  loc=1)
        ax.set_title('RNN Testing: Wait')
        ax.set_xlim([0, hp['dt'] * len(inp)])
        ax.set_ylim([-1.2, 1.2])
        ax.set_xlabel('Time (s)')
        test_fig.canvas.draw()

        E_out = 0  # Running squared error
        V_targ = 0  # Running variance of target
        print('Testing: %g trials' % hp['test_trials'])
        for idx in range(inp.shape[0]):
            trial = {
                'inp': inp[idx],
                'targ': targ[idx],
            }
            print('.', end="")

            tvec = np.arange(0, inp.shape[1]) * hp['dt']
            ax.set_xlim([0, hp['dt'] * inp.shape[1]])
            line_inp.set_xdata(tvec)
            line_inp.set_ydata(trial['inp'])
            line_targ.set_xdata(tvec)
            line_targ.set_ydata(trial['targ'])
            out = self.run(inp)[0]
            line_out.set_xdata(tvec)
            line_out.set_ydata(out)
            ax.set_title('RNN Testing, trial %g' % (idx + 1))
            test_fig.canvas.draw()

            E_out = E_out + (out - trial['targ']).T @ (out - trial['targ'])
            V_targ = V_targ + trial['targ'].T @ trial['targ']
        print('')
        E_norm = E_out / V_targ
        print('Normalized error: %g' % E_norm)
        return E_norm
