#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

from loaddata import load_dataset

from realtimeplot import Plot as Plotter
import matplotlib.pyplot as plt

from math import sqrt

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne

# Setup variables
out = 'out/autoencoder/' + str(int(time.time())) +'/'
Plot = False

# Plotting variables
if Plot:
    loss_plot = Plotter('Loss', 'epoch', 'loss')
epochs = []
training_losses = []
validation_losses = []
resultImgs = [None for x in range(6)] # 6 images
resultImgsFigId = 12
weightImgs = [None for x in range(100)] # 100 images
weightImgsFigId = 13

# Timing variables
times = []

# Hyperparams
initialLearningRate = 2.3
initialMomentum = 0.9
decay = 0.88
ratioPercentDiffThreshold = 8.0

# Theano variables
input_var = T.tensor4('inputs')

srng = RandomStreams(seed=234)

class ReshapeLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input.reshape(input_var.shape)

class NoiseLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        noise = srng.binomial(input.shape, n=1.0, p=0.7, dtype='float32')
        return input * noise

def iterate_minibatches(inputs, batchsize, shuffle=False):
    inputs = inputs.get_value()
    if batchsize >= inputs.shape[0]:
        yield inputs
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs), batchsize):
        end_idx = start_idx + batchsize
        if end_idx > len(inputs):
            end_idx = len(inputs)
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield inputs[excerpt]

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterations  - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        print("\n")


def main(num_epochs=50, batch_size=5000):
    # Load the dataset
    print("Loading data...")
    X_train, X_val, X_test = load_dataset()
    X_train = theano.shared(lasagne.utils.floatX(X_train))
    X_val = theano.shared(lasagne.utils.floatX(X_val))
    X_test = theano.shared(lasagne.utils.floatX(X_test))
    
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)

    # Add random binomial noise to the input image
    l_in_noise = NoiseLayer(l_in)

    # Add a fully-connected layer of 588 units using leaky ReLU and weights from a He Normal distribution
    # Note: using a *non* leaky ReLU resulted in drastically reduced performance!
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_noise, num_units=588,
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.HeUniform(gain=sqrt(2 / (1 + 0.01**2))))

    # We'll now add dropout of 20%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.2)

    # Finally, we'll add the fully-connected output layer.
    # The weights are regularized with the restriction of being the transpose of W0
    l_out = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=28 * 28,
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=l_hid1.W.T)

    # Reshape the output to a square for easy processing
    network = ReshapeLayer(l_out)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(input_var, prediction)
    loss = loss.mean()

    # Create a shared Theano variable for the learning rate that we can decay
    # as we iterate through epochs
    eta = theano.shared(np.array(initialLearningRate, dtype=theano.config.floatX))

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=eta, momentum=initialMomentum)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    noisyInput, l_hid1Activation, test_prediction = lasagne.layers.get_output([l_in_noise, l_hid1, network], deterministic=True)
    test_loss = lasagne.objectives.squared_error(input_var, test_prediction)
    test_loss = test_loss.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var], [loss], updates=updates)

    # Compile a second function computing the validation loss and intermediate outputs:
    val_fn = theano.function([input_var], [test_loss, noisyInput, l_hid1Activation, prediction])

    # Finally, launch the training loop.
    print("Starting training...")

    printProgress(0, num_epochs, prefix = 'Iterations', suffix = 'Complete', barLength = 50)

    training_start_time = time.time()
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # every 5 epochs, half our learning rate
        #if epoch % decayInterval == 0 and epoch > 0:
        #    eta.set_value(eta.get_value() * decay)

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        medianWeightUpdateRatio = []
        for batch in iterate_minibatches(X_train, batch_size, shuffle=True):
            inputs = batch

            # Train to get the loss and calculate the change in weights
            wBefore = l_hid1.W.get_value()
            err = train_fn(inputs)
            wAfter = l_hid1.W.get_value()
            u = wAfter - wBefore

            # Calculate the weight/update ratio
            param_scale = np.linalg.norm(wBefore.ravel())
            update_scale = np.linalg.norm(u.ravel())
            medianWeightUpdateRatio.append(update_scale / param_scale)

            train_err += err[0]
            train_batches += 1

        # Plot some of the first-layer weights
        if Plot:
            w = l_hid1.W.get_value()
            vals, num = w.shape
            f = plt.figure(weightImgsFigId)
            for i in range(len(weightImgs)):
                a = f.add_subplot(10, 10, i + 1)
                if weightImgs[i] is None:
                    weightImgs[i] = plt.imshow(w[:, i].reshape((28, 28)), cmap='Greys')
                else:
                    weightImgs[i].set_data(w[:, i].reshape((28, 28)))
            f.canvas.draw()

        # Calculate the *median* ratio and its percent difference from our target ratio, 1e-3
        avgRatio = np.median(medianWeightUpdateRatio)
        avg = avgRatio + 1e-3 / 2.0
        percentDiff = abs(avgRatio - 1e-3) / avg * 100.0

        # Check if the percent difference from 1e-3 is greater than a threshold.
        # If so, we need to change the learning rate.
        # Note that we ignore this for the first two epochs as learning is too steep, and it shouldn't
        # matter if you have a good starting learning rate.
        if percentDiff > ratioPercentDiffThreshold and epoch >= 2:
            old = eta.get_value() * 1.0
            if avgRatio > 1e-3: # too high
                new = old * decay
            else:
                new = old * (2.0 - decay)
            print('Warning: {:6.6f} is too far from 1.0'.format(avgRatio * 1e3))
            print('Changed learning rate from {:6.6f} to {:6.6f} because of a {:6.6f} perc. diff.' \
                .format(old, new, percentDiff))
            eta.set_value(lasagne.utils.floatX(new))

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, batch_size, shuffle=False):
            inputs = batch
            err, noiseInput, hidden1act, reconstructed = val_fn(inputs)
            val_err += err
            val_batches += 1

            if Plot:
                # Plot a histogram of hidden layer activations
                hist, bins = np.histogram(hidden1act[0], \
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, \
                    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], density=True)

                width = 0.7 * (bins[1] - bins[0])
                center = (bins[:-1] + bins[1:]) / 2
                fig = plt.figure(123)
                plt.title('Hidden Layer Activations')
                plt.gca().cla()
                plt.bar(center, hist, align='center', width=width)
                fig.canvas.draw()

                # Plot our original image, noisy image, and reconstructed image
                orig = inputs[0].reshape((28, 28))
                noise = noiseInput[0].reshape((28, 28))
                diff = orig - noise
                reconstructed = reconstructed[0].reshape((28, 28))
                fixedDiff =  reconstructed - noise

                imgs = [orig, diff, noise, noise, fixedDiff, reconstructed]
                titles = ['Original', 'Pixels Removed', 'Noisy', \
                        'Original', 'Pixels Added', 'Reconstructed']

                fig = plt.figure(resultImgsFigId)
                for i in range(len(resultImgs)):
                    a = fig.add_subplot(2,3,i + 1)
                    if resultImgs[i] is None:
                        resultImgs[i] = plt.imshow(imgs[i], cmap='Greys')
                    else:
                        resultImgs[i].set_data(imgs[i])
                    a.set_title(titles[i])
                fig.canvas.draw()

        if Plot:
            # Keep track of historical losses for our loss plot. Update the plot too.
            epochs.append(epoch)
            training_losses.append(train_err / float(train_batches))
            validation_losses.append(val_err / float(val_batches))
            loss_plot.update([epochs, epochs], [training_losses, validation_losses], ['train', 'val'])

        end_time = time.time()
        times.append(end_time - start_time)

        printProgress(epoch + 1, num_epochs, prefix = 'Iterations', suffix = 'Complete', barLength = 50)

        # Then we print the results for this epoch:
        #print("Epoch {} of {} took {:.3f}s".format(
        #    epoch + 1, num_epochs, end_time - start_time))
        #print("  learning rate:\t\t{0}".format(eta.get_value()))
        #print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        #print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

    # After training, we compute and print the test error:
    test_err = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, batch_size, shuffle=False):
        inputs = batch
        err, _, _, _ = val_fn(inputs)
        test_err += err
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  average epoch time:\t\t{:.6f}".format(np.mean(times)))
    print("  total runtime:\t\t{:.6f}".format(time.time() - training_start_time))

    # Write our charts out and some data about the run.
    #if not os.path.exists(out):
    #    os.makedirs(out)

    if Plot:
        loss_plot.save(out)
        f = plt.figure(resultImgsFigId)
        f.savefig(out + 'results.png')
        f = plt.figure(weightImgsFigId)
        f.savefig(out + 'weights.png')

    #f = open(out + 'log.txt', 'w')
    #f.write('Epochs:\t\t' + str(epoch))
    #f.write('\nInitial learning rate:\t\t' + str(initialLearningRate))
    #f.write('\nInitial momentum:\t\t' + str(initialMomentum))
    #f.write('\nFinal test loss:\t\t' + str((test_err / test_batches)))
    #f.close()

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [EPOCHS] [BATCH_SIZE]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
            kwargs['batch_size'] = int(sys.argv[2])
        main(**kwargs)
