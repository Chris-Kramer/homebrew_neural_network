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
# Kernel
from utils import create_1d_kernel
# Feed forward weights
from utils import init_weight
from utils import init_bias
# data
from utils import gen_cnn_test_data, gen_cnn_train_data
from utils import get_rand_sample


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

def cnn(data, labels, epochs, learning_rate, batch_size):
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

                # ----- Conv1D -----
                filter_1 = conv1D(data_sample, kernel_1, 3)
                filter_2 = conv1D(data_sample, kernel_2, 3)

                # ----- Activation -----
                for i in range(len(filter_1)):
                    filter_1[i] = relu_function(filter_1[i])

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
            loss = cross_entropy_loss(batch_preds, batch_obs)

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
            weight_1 = stoch_grad_desc(weight_1, weight_1_grad, learning_rate, decay_rate, n_epoch)
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


#############
# Run model #
#############

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

##############
# Plot model #
# ############
# Plot loss
plt.plot(train_epoch, train_loss, label = "Training")
plt.plot(test_epoch, test_loss, label = "Test")
plt.title("Avg. loss pr. epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()  

# Plot accuracy
plt.plot(train_epoch, train_acc, label = "Training")
plt.plot(test_epoch, test_acc, label = "Test")
plt.title("Accuracy pr. epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()  
