
from math import e
from math import log
from statistics import mean
from random import uniform
from random import randint
import pandas as pd
import tensorflow as tf
##################
# Ouput and loss #
##################
def calc_accuracy(preds, true_vals):
    """
    Calculate the accuracy between list of predictions and true values
    """
    n_correct = 0
    for pred, true_val in zip(preds, true_vals):
        pred = convert_to_label(pred)
        if int(pred) == int(true_val):
            n_correct += 1
    return (n_correct / len(true_vals)) * 100.0

def convert_to_label(pred):
    """
    Convert sigmoid prediction to a label
    """
    if pred <= 0.5:
        return 0
    else:
        return 1

def cross_entropy_loss(preds, true_vals):
    """
    Binary cross entropy loss functions for the batch (mean)
    """
    results = []
    for pred, true_val in zip(preds, true_vals):
        if true_val == 1:
            result = -log(pred)
        else:
            result = -log( 1 - pred)
        results.append(result)
    return mean(results)

###############
# Activations #
###############
def relu_function(x):
    """
    ReLU hidden activation function
    """
    return max(0, x)

def sigmoid_function(x):
    
    """
    Sigmoid output activation function
    """
    res = ( 1.0 )/( 1.0 + e**(-x) )
    if res == 1: # happens due to rounding
        res = 0.9999999999
    elif res == 0: # happens due to rounding
        res = 0.0000000001
    return res

###############################
# Stochastic Gradient descent #
###############################
past_velocity = 0
def stoch_grad_desc(weight, gradient, init_LR, decay, epoch, momentum = 0.01):
    """
    STG with momentum and an exponential decaying LR
    """
    global past_velocity
    learning_rate = init_LR * e**(-decay * epoch) # Keras exponential decay rate formula
    velocity = (past_velocity * momentum) - (learning_rate * gradient)
    weight = (weight + (momentum * velocity)) - (learning_rate * gradient)
    past_velocity = velocity
    return weight

#############
# Gradients #
#############
def calc_loss_gradient(preds, true_vals):
    """
    Calculation gradient of loss function
    """
    results = []
    for pred, true_val in zip(preds, true_vals):
        result = ((-true_val)/pred) + (1 - true_val)/(1 - pred)
        results.append(result)
    return mean(results)

def calc_sigmoid_gradient(preds):
    """
    Calculate gradient of sigmoid function
    """
    results = []
    for pred in preds:   
        result = pred * (1 - pred)
        results.append(result)
    return mean(results)

def calc_relu_gradient(x):
    """
    Calculate gradient of relu function
    """
    return 0 if x <= 0 else 1

def calc_max_pool_grads(filter_list, upstream_gradient):
    """
    Calculate gradients of max pool operation
    """
    max_pool_grads = []
    for val in filter_list:
        if round(val, 10) == max(filter_list,):
            max_pool_grads.append(1 * upstream_gradient)
        else:
            max_pool_grads.append(0 * upstream_gradient)
    return max_pool_grads

################
# Convolutions #
################
def _pad_list(input_list, size):
    """
    Applies padding to input list
    """
    # Apply padding to list
    while len(input_list) < size:
        input_list.append(0)
    return input_list

def _divide_list(input_list, size, strides):
    """
    Divide list into equal sized chunks
    Applies padding if not equal length
    """
    divided_list = []
    for i in range(0, len(input_list), strides):
        x = i
        small_list = input_list[x : x+ size]
        small_list = _pad_list(small_list, size)
        divided_list.append(small_list)
    return divided_list

# Convolution
def conv1D(input_list, kernel, strides):
    """
    Perform a 1d convolution on input list with kernel
    """
    divided_list = _divide_list(input_list, len(kernel), strides)
    # Gets dot product of the kernel and the sub lists
    return_list = []
    for sub_list in divided_list:
        temp_list = []
        for list_val, kernel_val in zip(sub_list, kernel):
            temp_list.append(list_val * kernel_val)
        return_list.append(sum(temp_list))
    return return_list

###########
# Pooling #
###########
def max_pool(input_list, pool_size, strides):
    """
    Perform max pooling on list
    """
    divided_list = _divide_list(input_list, pool_size, strides)
    return_list = []
    for sub_list in divided_list:
        return_list.append(max(sub_list))
    return return_list


def avg_pool(input_list, pool_size, strides):
    """
    Performs avg. pooling on list
    """
    divided_list = _divide_list(input_list, pool_size, strides)
    return_list = []
    for sub_list in divided_list:
        return_list.append(mean(sub_list))
    return return_list

##########
# Kernel #
##########
def create_1d_kernel(kernel_length):
    """
    Creates a 1d kernel
    Contains random floats between 0 and 1
    """
    return [uniform(0, 1) for _ in range(kernel_length)]

######################
# Initialize weights #
######################
def init_weight():
    """
    Returns weigth with random float
    Between 0 and 1
    """
    return uniform(0, 1)

def init_bias():
    """
    Returns bias
    """
    return 1

#########################
# Generate data for cnn #
#########################
def gen_cnn_train_data(n_samples):
    """
    Generate data of weekly spending for training
    This is very structured data
    """
    data = []
    labels = []

    normal_spend = [100, 200, 300, 400, 500]
    vacation_spend = [400, 500, 600, 700, 800]
    for _ in range(n_samples):
        normal_week = [normal_spend[randint(0,4)] for i in range(7)]
        data.append(normal_week)
        labels.append(0)

        vacation_week = [vacation_spend[randint(0,4)] for i in range(7)]
        data.append(vacation_week)
        labels.append(1)
    return (data, labels)

def gen_cnn_test_data(n_samples):
    """
    Generate data of weekly spending for test
    This is unstructured
    """
    data = []
    labels = []

    for _ in range(n_samples):
        normal_week = [randint(100,500) for i in range(7)]
        data.append(normal_week)
        labels.append(0)

        vacation_week = [randint(250,900) for i in range(7)]
        data.append(vacation_week)
        labels.append(1)
    return (data, labels)

def get_rand_sample(data, labels = None):
    """
    Returns a random sample of data and label
    """
    select = randint(0, len(data) - 1)
    return (data[select], labels[select]) 

def load_train_data():
    """
    Returns the training data and labels as list of lists
    For the feed forward network
    """
    # Data
    data = pd.read_csv("tickets_train.csv")
    data = data.reset_index(drop=True)
    train_data = data[["temp", "week", "rank", "price"]]
    train_data = train_data.values.tolist()
    # labels
    labels = data["buy"]
    labels.values.tolist()
    return (train_data, labels)

def load_test_data():
    """
    Loads the test data
    For the feed forward network
    """
    data = pd.read_csv("tickets_test.csv")
    test_data = data.reset_index(drop=True)
    test_data = data[["temp", "week", "rank", "price"]]
    test_data = test_data.values.tolist()
    # labels
    labels = data["buy"]
    labels.values.tolist()
    return (test_data, labels)

##################
# Regularization #
##################
def _split_data(data_list, label_list, size):
    """
    Divide list into equal sized chunks
    """
    divided_data = []
    divided_labs = []
    for i in range(0, len(data_list), size):
        x = i
        small_data_list = data_list[x : x+ size]
        small_labs_list = label_list[x : x+ size]
        divided_data.append(small_data_list)
        divided_labs.append(small_labs_list)
    return (divided_data, divided_labs)

def k_fold_cross(k, data, labels):
    """
    Splits the data into k chunks
    """
    X, Y =  _split_data(data, labels, size = len(data)//k)
    return (X, Y)


def L2_loss_pen(penalty, weights):
    """
    The loss penalty for L2
    """
    squared_w = []
    for w in weights:
        squared_w.append(w**2)
    return  penalty * sum(squared_w)

def L2_weight_pen(penalty, learning_rate):
    """
    The L2 weight penalty for updating
    """
    return (1 - learning_rate * penalty)

def L1_loss_pen(penalty, weights):
    """
    The L1 loss penalty
    """
    return  penalty * abs(sum(weights))

def dropout(prob):
    """
    Drops a hiddens unit if probability is reached
    """
    return uniform(0,1) >= prob

def scale_kernel(kernel, dropout_rate):
    """
    Scales the kernel weights by dropout rate
    """
    z_kern = []
    for i in range(len(kernel)):
        z = kernel[i] * (1/(1- dropout_rate))
        z_kern.append(z)
    return z_kern

def gen_synth_data(data):
    """
    Generates synthetic data
    Based on existing data
    It just reversed existing data
    """
    synth_data = []
    for sample in data:
        synth_sample = reversed(sample)
        synth_data.append(synth_sample)
        synth_data.append(sample)
    return synth_data

def inject_noise():
    """
    Injects noise to unit
    """
    return uniform(0, 1)


def _sample_with_replace(x_train, y_train, k):
    """
    Bootstrap sampling with replacement
    """
    # Lists with k data sets
    x_k_samples = []
    y_k_samples = []
    n = len(x_train)
    for i in range(k):

        # Get random samples
        x_sub_samples = []
        y_sub_samples = []
        for j in range(n - 1):
            select = randint(0, n - 1)
            x_sub_samples.append(x_train[select])
            y_sub_samples.append(y_train[select])

        x_k_samples.append(x_sub_samples)
        y_k_samples.append(y_sub_samples)
    return (x_k_samples, y_k_samples)

def _train_sample_models(model, x_samples, y_samples, epochs = 10, batch_size = 128):
    """
    Trains k models on data which have been k sampled
    """
    models = []
    for x_train, y_train in zip(x_samples, y_samples):
        x_train = tf.convert_to_tensor(x_train)
        y_train = tf.convert_to_tensor(y_train)
        model.fit(x_train, y_train, verbose = False, epochs = epochs, batch_size = batch_size)
        models.append(model)
        model.reset_states()
    return models

def _make_models_preds(models, x_test):
    """
    Return predictions from a list of models
    """
    models_preds = []
    for m in models:
        m_preds = m.predict(x_test)
        models_preds.append(m_preds)
    return models_preds

def _get_avg_preds(models_preds, models):
    """
    Average the predictions from a list of models and predictions
    """
    n = len(models_preds[0])
    final_preds = []
    for i in range(n):
        models_pred = []
        for m in range(len(models)):
            m_pred = models_preds[m][i][0]
            models_pred.append(m_pred)
        final_preds.append(mean(models_pred))
    return final_preds

def bootstrap_agg(x_train, y_train, x_test, model, k, epochs = 10, batch_size = 128):
    """
    Perform bootstrap aggregation
    """
    x_samples, y_samples = _sample_with_replace(x_train, y_train, k)
    models = _train_sample_models(model, x_samples, y_samples, epochs, batch_size)
    models_preds = _make_models_preds(models, x_test)
    return _get_avg_preds(models_preds, models)

###########################
# Boring helper functions #
###########################
def switch_places(a, b, l):
    """
    Switches places between two list elements by index
    """
    l[a], l[b] = l[b], l[a]
    return l