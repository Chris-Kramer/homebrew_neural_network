
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution1D, Dropout, MaxPool1D, AveragePooling1D
from keras.optimizers import SGD
from keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import gen_cnn_test_data, gen_cnn_train_data
from utils import load_test_data, load_train_data
from utils import bootstrap_agg
from utils import cross_entropy_loss
from utils import calc_accuracy

########
# Data #
########
# Training data
X_train, Y_train = gen_cnn_train_data(200)
x_train = []
y_train = []
for x, y in zip(X_train, Y_train):
    x_train.append(tf.convert_to_tensor(x))
    y_train.append(tf.convert_to_tensor(y))
x_train = tf.convert_to_tensor(x_train)
y_train = tf.convert_to_tensor(y_train)

# Test data
X_test, Y_test = gen_cnn_test_data(200)
x_test = []
y_test = []
for x, y in zip(X_test, Y_test):
    x_test.append(tf.convert_to_tensor(x))
    y_test.append(tf.convert_to_tensor(y))

x_test = tf.convert_to_tensor(x_test)
y_test = tf.convert_to_tensor(y_test)

###############
# Hyperparams #
###############
learning_rate = 0.001
momentum = 0.01

##########
# Models #
##########
"""
# CNN model that underfits like mine
model = Sequential()
model.add(Convolution1D(filters = 2, kernel_size=3, activation="relu", input_shape = (x_train.shape[1],1)))
model.add(MaxPool1D())
model.add(Dense(1, activation="relu"))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
sgd = SGD(lr=learning_rate, momentum=momentum)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])
history = model.fit(x_train, y_train, validation_data= (x_test, y_test), epochs=10, batch_size=128)
"""
# Bootstrap model
# CNN model that underfits like mine
underfit_model = Sequential()
underfit_model.add(Convolution1D(filters = 2, kernel_size=3, activation="relu", input_shape = (x_train.shape[1],1)))
underfit_model.add(MaxPool1D())
underfit_model.add(Dense(1, activation="relu"))
underfit_model.add(Flatten())
underfit_model.add(Dense(1, activation="sigmoid"))
sgd = SGD(lr=learning_rate, momentum=momentum)
underfit_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])


final_preds = bootstrap_agg(x_train, y_train, x_test, underfit_model, 10)
print("Underfit Model")
print(cross_entropy_loss(final_preds, y_test))
print(calc_accuracy(final_preds, y_test))

overfit_model = Sequential()
overfit_model.add(Dense(250, input_shape=(7,), activation="relu"))
overfit_model.add(Dense(250, activation="relu"))
overfit_model.add(Dense(128, activation="relu"))
overfit_model.add(Dense(128, activation="relu"))
overfit_model.add(Dense(64, activation="relu"))
overfit_model.add(Dense(64, activation="relu"))
overfit_model.add(Dense(32, activation="relu"))
overfit_model.add(Dense(32, activation="relu"))
overfit_model.add(Dense(16, activation="relu"))
overfit_model.add(Dense(16, activation="relu"))
overfit_model.add(Dense(1, activation="sigmoid"))
sgd = SGD(lr=learning_rate, momentum=momentum)
overfit_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])

final_preds = bootstrap_agg(x_train, y_train, x_test, overfit_model, 10, epochs= 500)
print("Overfit Model")
print(cross_entropy_loss(final_preds, y_test))
print(calc_accuracy(final_preds, y_test))
"""
# Model that overfits
model = Sequential()
model.add(Dense(250, input_shape=(7,), activation="relu"))
model.add(Dense(250, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
sgd = SGD(lr=learning_rate, momentum=momentum)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])
history = model.fit(x_train, y_train, validation_data= (x_test, y_test), epochs=500, batch_size=128)

# Plot model
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Accuracy pr. epoch")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Avg. loss pr. epoch")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Traning', 'Test'], loc='upper left')
plt.show()
"""
"""
X_train, Y_train = load_train_data()
X_test, Y_test = load_test_data()
x_train = []
y_train = []
for x, y in zip(X_train, Y_train):
    x_train.append(tf.convert_to_tensor(x))
    y_train.append(tf.convert_to_tensor(y))

x_train = tf.convert_to_tensor(x_train)
y_train = tf.convert_to_tensor(y_train)
x_test = []
y_test = []
for x, y in zip(X_test, Y_test):
    x_test.append(tf.convert_to_tensor(x))
    y_test.append(tf.convert_to_tensor(y))

x_test = tf.convert_to_tensor(x_test)
y_test = tf.convert_to_tensor(y_test)

learning_rate = 0.009
momentum = 0.01
print(x_train.shape)
model = Sequential()
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
sgd = SGD(lr=learning_rate, momentum=momentum)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])

history = model.fit(x_train, y_train, validation_data= (x_test, y_test), epochs=10, batch_size=128)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Accuracy pr. epoch")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Avg. loss pr. epoch")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""

"""
# #########
# Imports #
###########
# Plotting
from matplotlib import pyplot as plt
# Acitvations
from utils import relu_function
from utils import sigmoid_function
# Output
from utils import calc_accuracy
from utils import convert_to_label
from utils import cross_entropy_loss
# Gradients
from utils import calc_loss_gradient
from utils import calc_sigmoid_gradient
from utils import calc_relu_gradient
from utils import calc_max_pool_grads
# STG
from utils import stoch_grad_desc
# Convolutions
from utils import conv1D
from utils import max_pool
from utils import avg_pool
# Kernel
from utils import create_1d_kernel
# Feed forward weights
from utils import init_weight
from utils import init_bias
# data
from utils import gen_cnn_test_data, gen_cnn_train_data
from utils import get_rand_sample
from utils import k_fold_cross
# Regularization
from utils import L2_weight_pen, L2_loss_pen
from utils import dropout, scale_kernel
from utils import inject_noise
# Helper functions
from utils import switch_places
from statistics import mean

#########
# Model #
#########
# ----- Initilialze -----
# Outside scope to save weigths
# kernels
kernel_1 = create_1d_kernel(3)
kernel_2 = create_1d_kernel(3)

# Feed forward
weight_1 = init_weight()
weight_2 = init_weight()
bias = init_bias()

def cnn(data, labels, epochs, learning_rate, batch_size, early_stop = 4):
    # Trainable params
    global kernel_1
    global kernel_2
    global weight_1
    global weight_2
    global bias

    #Statistics
    loss_hist = []
    epoch_hist = []
    acc_hist = []

    n_loss_rise = 0
    prev_loss = None
    # ----- Run epoch -----
    for epoch in range(epochs):
        n_epoch = epoch + 1
        epoch_preds = []
        epoch_obs = []

        for _ in range(len(data)):
            batch_preds = []
            batch_obs = [] 
            
            # ----- Mini batch -----
            for i in range(batch_size): 
                data_sample, true_val = get_rand_sample(data, labels)

                # ----- Conv1D Dropout -----
                dropout_rate = 0.5
                drop_output = [0,0,0]

                # If kernel 1 is dropped
                if dropout(dropout_rate):
                    kernel_2 = scale_kernel(kernel_2, dropout_rate)
                    filter_1 = conv1D(data_sample, drop_output, 3)
                    filter_2 = conv1D(data_sample, kernel_2, 3)
            
                # Else If kernel 2 is dropped
                elif dropout(dropout_rate):
                    kernel_1 = scale_kernel(kernel_1, dropout_rate)
                    filter_1 = conv1D(data_sample, kernel_1, 3)
                    filter_2 = conv1D(data_sample, drop_output, 3)                
                
                # No dropout
                else:
                    filter_2 = conv1D(data_sample, kernel_1, 3)    
                    filter_2 = conv1D(data_sample, kernel_2, 3)
            

                # ----- Activation -----
                for i in range(len(filter_1)):
                    filter_1[i] = relu_function(filter_1[i] * inject_noise())

                for i in range(len(filter_2)):
                    filter_2[i] = relu_function(filter_2[i])
                
                # ----- Max pooling -----
                filter1_max_pool = max_pool(filter_1, pool_size = 3, strides = 3)
                filter2_max_pool = max_pool(filter_2, pool_size = 3, strides = 3)

                # ----- Feed forward input ------
                input1 = filter1_max_pool[0]
                input2 = filter2_max_pool[0]

                # ----- Feed forward hidden -----
                weight_res = ((input1 * weight_1) + (input2 * weight_2)) + bias
                activation = relu_function(weight_res)

                # ----- Feed forward Output -----
                pred = sigmoid_function((activation))

                batch_preds.append(pred)
                batch_obs.append(true_val)
            
            # ----- Loss -----
            weigths = [weight_1, weight_2]
            for k1_w, k2_w in zip(kernel_1, kernel_2):
                weigths.append(k1_w)
                weigths.append(k2_w)

            loss = cross_entropy_loss(batch_preds, batch_obs) + L2_loss_pen(0.01, weigths)
            
            n_loss_rise += 1 if loss > prev_loss else n_loss_rise
            if n_loss_rise >= early_stop:
                return (epoch_hist, loss_hist, acc_hist)
            
            # ----- Back propagation -----
            # Loss gradient --> sigmoid
            local_loss_grad = calc_loss_gradient(batch_preds, batch_obs)
            first_upstream_grad = local_loss_grad

            # Sigmoid gradient --> Relu
            local_sigmoid_grad = calc_sigmoid_gradient(batch_preds)
            second_upstream_grad = first_upstream_grad * local_sigmoid_grad

            # ReLu gradient --> Affine transformation
            local_relu_grad = calc_relu_gradient(weight_res)
            third_upstream_grad = second_upstream_grad * local_relu_grad 

            # Affine transformation gradient  --> weights and bias 
            weight_1_grad = third_upstream_grad * weight_1 # local gradient is just the weight
            weight_2_grad = third_upstream_grad * weight_2 # local gradient is just the weight 
            bias_grad = third_upstream_grad * 1 # Local gradient is 1 since it is addition to function
            fourth_upstream_grad_k1 =  weight_1_grad
            fourth_upstream_grad_k2 = weight_2_grad

            # Maxpool gradients --> kernel 1
            max_pool_k1_grads = calc_max_pool_grads(filter_1, fourth_upstream_grad_k1)

            # Max pool gradients --> kernel 2
            max_pool_k2_grads = calc_max_pool_grads(filter_2, fourth_upstream_grad_k2)

            # Kernel 1 gradients
            kernel_1_grads = conv1D(data_sample, max_pool_k1_grads, 3)

            # Kernel 2 gradients
            kernel_2_grads = conv1D(data_sample, max_pool_k2_grads, 3)
        
            # Stochastic gradient descent
            decay_rate = learning_rate / epochs
            
            # Feed forward input
            weight_1 = L2_weight_pen(0.01, weight_1) * weight_1
            weight_1 = stoch_grad_desc(weight_1, weight_1_grad,
                                       learning_rate,decay_rate, n_epoch)


            weight_2 = stoch_grad_desc(weight_2, weight_2_grad, learning_rate, decay_rate, n_epoch)
            bias = stoch_grad_desc(bias, bias_grad, learning_rate, decay_rate, n_epoch)

            # Kernel 1
            for i in range(len(kernel_1)):
                kernel_1[i] = stoch_grad_desc(kernel_1[i], kernel_1_grads[i], learning_rate, decay_rate, n_epoch)

            # Kernel 2
            for i in range(len(kernel_2)):
                kernel_2[i] = stoch_grad_desc(kernel_2[i], kernel_2_grads[i], learning_rate, decay_rate, n_epoch)

            epoch_obs.append(true_val)
            epoch_preds.append(convert_to_label(pred))
                
        # ----- End of epoch -----
        epoch_acc = calc_accuracy(epoch_preds, epoch_obs)
        print(f"######## Epoch: {n_epoch} ######## ")
        print(f"Loss: {loss}, accuracy: {epoch_acc}")
        acc_hist.append(epoch_acc)
        loss_hist.append(loss)
        epoch_hist.append(epoch)
    
    # ----- End of training -----
    return (epoch_hist, loss_hist, acc_hist)


####################
# Cross validation #
####################
from statistics import mean
from utils import switch_places
from utils import k_fold_cross
#----- Hyperparams -----
epochs = 10
learning_rate = 0.001
batch_size = 128

# ----- Training ----
train_data, train_labels = gen_cnn_train_data(200)
train_epoch, train_loss, train_acc = cnn(train_data, train_labels, epochs, learning_rate, batch_size)

# ----- Test -----
test_data, test_labels = gen_cnn_test_data(200)
test_epoch, test_loss, test_acc = cnn(test_data, test_labels, epochs, learning_rate, batch_size) 

train_avg_loss = []
test_avg_loss = []
test_avg_acc = []
train_avg_acc = []

# Split data into folds
split_x, split_y = k_fold_cross(k = 10, data = train_data + test_data,
                                labels = train_labels + test_labels)
counter = 0
while counter <= len(split_x) - 1:
    # Reinit weights and biases
    kernel_1 = create_1d_kernel(3)
    kernel_2 = create_1d_kernel(3)
    weight_1 = init_weight()
    weight_2 = init_weight()
    bias = init_bias()

    # Get train labels
    x_train = split_x[0: len(split_x) -2]
    y_train = split_y[0: len(split_x) -2]
    # Get test labels
    x_test = split_x[-1]
    y_test = split_y[-1]
    # Move new test labels to end of list
    split_x = switch_places(counter, -1, split_x)
    split_y = switch_places(counter, -1, split_y)
    # Train models
    train_epoch, train_loss, train_acc = cnn(train_data, train_labels, 10,
                                             learning_rate, batch_size)
    test_epoch, test_loss, test_acc = cnn(test_data, test_labels, 10,
                                          learning_rate, batch_size)
    train_avg_loss.append(mean(train_loss)) 
    test_avg_loss.append(mean(test_loss))
    train_avg_acc.append(mean(train_acc))
    test_avg_acc.append(mean(test_acc))
    counter += 1

# Plot loss
width = 0.40
x_1 = [i + 0.2 for i in range(counter)]
x_2 = [i - 0.2 for i in range(counter)]
plt.bar(x_1, train_avg_loss, width = width, label = "Training")
plt.bar(x_2, test_avg_loss, width = width, label = "Test")
plt.title("Avg. loss pr. model")
plt.xlabel("Model")
plt.ylabel("Loss")
plt.xticks(range(counter))
plt.legend()
plt.show()  

# Plot accuracy
plt.bar(x_1, train_avg_acc, width = width, label = "Training")
plt.bar(x_2, test_avg_acc, width = width, label = "Test")
plt.title("Accuracy pr. Model")
plt.xlabel("Model")
plt.xticks(range(counter))
plt.ylabel("Accuracy")
plt.legend()
plt.show()

"""
