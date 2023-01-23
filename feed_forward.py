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
# STG
from utils import stoch_grad_desc
# data
from utils import get_rand_sample
from utils import load_train_data
from utils import load_test_data
import pandas as pd
# Feed forward weights
from utils import init_weight
from utils import init_bias

#########
# Model #
#########
# ------ Initialize -----
# Outside scope to save values during training
week_weight = init_weight()
temp_weight = init_weight()
rank_weight = init_weight()
price_weight = init_weight()
input_bias = init_bias()

def feed_forward(data, labels, learning_rate, epochs, batch_size):
    # Trainable parameters
    global week_weight
    global temp_weight
    global rank_weight
    global price_weight
    global input_bias

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
            # Mini batch
            for i in range(batch_size):

                # ----- Input Layer -----
                sample_data, true_val = get_rand_sample(data, labels)

                temp_input = sample_data[0]
                week_input = sample_data[1]
                rank_input = sample_data[2]
                price_input = sample_data[3]

                # ----- Hidden Layer -----
                weight_res = (((temp_input * temp_weight) +
                                (week_input * week_weight) +
                                (rank_input * rank_weight) + 
                                (price_input * price_weight)) + input_bias)

                activation = relu_function(weight_res)
                
                # ----- Output layer -----
                pred = sigmoid_function((activation))

                batch_preds.append(pred)
                batch_obs.append(true_val)
            
            # ----- loss ------
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

            # Affine transformation gradient --> weights and bias 
            # local gradients
            week_local_grad = week_weight
            temp_local_grad = temp_weight
            price_local_grad = price_weight
            rank_local_grad = rank_weight
            bias_local_grad = 1

            # Final gradients
            week_grad = third_upstream_grad * week_local_grad
            temp_grad = third_upstream_grad * temp_local_grad
            price_grad = third_upstream_grad * price_local_grad
            rank_grad = third_upstream_grad * rank_local_grad
            bias_grad = third_upstream_grad * bias_local_grad

            # ----- Stochastic Gradient Descent -----
            decay_rate = learning_rate / epochs
            input_bias = stoch_grad_desc(input_bias, bias_grad, decay_rate, n_epoch, learning_rate)
            temp_weight = stoch_grad_desc(temp_weight, temp_grad, decay_rate, n_epoch, learning_rate)
            price_weight = stoch_grad_desc(price_weight, price_grad, decay_rate, n_epoch, learning_rate)
            week_weight = stoch_grad_desc(week_weight, week_grad, decay_rate, n_epoch, learning_rate)
            rank_weight = stoch_grad_desc(rank_weight, rank_grad, decay_rate, n_epoch, learning_rate)

            # ----- Update stats -----
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

# ----- Hyperparams ------
batch_size = 128
epochs = 10
learning_rate = 0.1

# ------ Training -----
train_data, train_labels = load_train_data()
epoch_train, loss_hist_train,acc_hist_train = feed_forward(train_data, train_labels, learning_rate, epochs, batch_size)

# ----- Test ------
test_data, test_labels = load_test_data()
epoch_test, loss_hist_test, acc_hist_test = feed_forward(test_data, test_labels, learning_rate, epochs, batch_size)

##############
# Plot model #
##############
# ----- Loss ------
plt.plot(epoch_train, loss_hist_train, label = "Avg. Epoch loss: Training")
plt.plot(epoch_test, loss_hist_test, label = "Avg. Epoch loss: Test")
plt.title("Avg. loss pr. epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()  

# ----- Accuracy -----
plt.plot(epoch_train, acc_hist_train, label = "Epoch Accuracy: Training")
plt.plot(epoch_train, acc_hist_test, label = "Epoch Accuracy: Test")
plt.title("Accuracy pr. epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()  
