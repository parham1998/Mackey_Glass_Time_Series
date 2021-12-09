# =============================================================================
# Import required libraries
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import timeit

# =============================================================================
# Definition and derivation of sigmoid
# =============================================================================
def sigmoid(x):
    return  1 /( 1 + (math.e)**(-1 * x))

def sigmoid_deriviate(x):
    a = sigmoid(x)
    a = np.reshape(a, (-1,1))
    b = 1 - sigmoid(x)
    b = np.reshape(b, (-1,1))
    b = np.transpose(b)
    return np.diag(np.diag(np.matmul(a,b)))

# =============================================================================
# Read and normalize data
# =============================================================================
data = pd.read_csv('data.csv', header=None)
data = np.array(data)

min = np.min(data)
max = np.max(data)
for i in range(np.shape(data)[0]):
    for j in range(np.shape(data)[1]):
        data[i,j] = (data[i,j] - min) / (max - min)
        
# =============================================================================
# Define train_set - validation_set - test_set
# =============================================================================
split_ratio_train = 0.7
split_ratio_validation = 0.25

split_line_number = int(np.shape(data)[0] * split_ratio_train)
x_train = data[:split_line_number, :3]
y_train = data[:split_line_number, 3]

other_data = data[split_line_number:, :4]
split_line_number = int(np.shape(data)[0] * split_ratio_validation)

x_validation = other_data[:split_line_number, :3]
y_validation = other_data[:split_line_number, 3]

x_test = other_data[split_line_number:, :3]
y_test = other_data[split_line_number:, 3]

# =============================================================================
# Define MLP
# =============================================================================
input_dimension = np.shape(x_train)[1]
l1_neurons = 5
l2_neurons = 1

np.random.seed(20)
w1 = np.random.uniform(low=-1, high=1, size=(input_dimension, l1_neurons))
w2 = np.random.uniform(low=-1, high=1, size=(l1_neurons, l2_neurons))

# =============================================================================
# Training
# =============================================================================
lambda1 = 0.005
lambda2 = 0.005
lr1 = np.random.uniform(low=0, high=0.5, size=(input_dimension, l1_neurons))
lr2 = np.random.uniform(low=0, high=0.5, size=(l1_neurons, l2_neurons))
epochs = 40

MSE_train = []
MSE_validation = []

def Train(w1, w2, lr1, lr2):
    output_train = []
    sqr_err_epoch_train = []
    ew_11 = 0
    ew_12 = 0
    for i in range(np.shape(x_train)[0]):
        x = np.reshape(x_train[i], (1,-1)) # x: (1, 3)
        # Feed-Forward
        # Layer 1
        net1 = np.matmul(x, w1) # net1: (1, 5)
        o1 = sigmoid(net1) # o1: (1, 5)
        # Layer 2
        net2 = np.matmul(o1, w2) # net2: (1, 1)
        o2 = net2 # net2: (1, 1)

        output_train.append(o2[0])

        # Error
        err = y_train[i] - o2[0]
        sqr_err_epoch_train.append(err**2)

        # Back propagation
        f_driviate = sigmoid_deriviate(net1) # f_driviate: (5, 5)
        w2_f_deriviate = np.matmul(f_driviate, w2) # w2_f_deriviate: (5, 1)
        w2_f_deriviate_x = np.matmul(w2_f_deriviate, x) # w2_f_deriviate_x: (5, 3)
        # first train w1 then w2
        ew1 = err * -1 * 1 * w2_f_deriviate_x
        lr1 = np.add(lr1 , np.transpose(lambda1 * ew1 * ew_11))
        w1 = np.subtract(w1 , lr1 * np.transpose(ew1))
        ew_11 = ew1
        #
        ew2 = err * -1 * 1 * np.transpose(o1)
        lr2 = np.add(lr2 , (lambda2 * ew2 * ew_12))
        w2 = np.subtract(w2 , lr2 * ew2)
        ew_12 = ew2

    mse_epoch_train = 0.5 * ((sum(sqr_err_epoch_train))/np.shape(x_train)[0])
    MSE_train.append(mse_epoch_train[0])
    return output_train, w1, w2, lr1, lr2

def Validation(w1, w2):
    sqr_err_epoch_validation = []
    output_validation = []
    
    for i in range(np.shape(x_validation)[0]):
        x = np.reshape(x_validation[i], (1,-1))
        # Feed-Forward
        # Layer 1
        net1 = np.matmul(x, w1)
        o1 = sigmoid(net1)
        # Layer 2
        net2 = np.matmul(o1, w2)
        o2 = net2

        output_validation.append(o2[0])

        # Error
        err = y_validation[i] - o2[0]
        sqr_err_epoch_validation.append(err ** 2)

    mse_epoch_validation = 0.5 * ((sum(sqr_err_epoch_validation))/np.shape(x_validation)[0])
    MSE_validation.append(mse_epoch_validation[0])
    return output_validation

def Plot_results(output_train, 
                 output_validation, 
                 m_train, 
                 b_train,
                 m_validation,
                 b_validation):
    # Plots
    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(15, 15)
    axs[0, 0].plot(MSE_train,'b')
    axs[0, 0].set_title('MSE Train')
    axs[0, 1].plot(MSE_validation,'r')
    axs[0, 1].set_title('Mse Validation')

    axs[1, 0].plot(y_train, 'b')
    axs[1, 0].plot(output_train,'r')
    axs[1, 0].set_title('Output Train')
    axs[1, 1].plot(y_validation, 'b')
    axs[1, 1].plot(output_validation,'r')
    axs[1, 1].set_title('Output Validation')

    axs[2, 0].plot(y_train, output_train, 'b*')
    axs[2, 0].plot(y_train, m_train*y_train+b_train,'r')
    axs[2, 0].set_title('Regression Train')
    axs[2, 1].plot(y_validation, output_validation, 'b*')
    axs[2, 1].plot(y_validation, m_validation*y_validation+b_validation,'r')
    axs[2, 1].set_title('Regression Validation')
    plt.show()
    time.sleep(1)
    plt.close(fig)
    
print('==> Start Training ...')
for epoch in range(epochs):    
    start = timeit.default_timer()
    
    output_train, w1, w2, lr1, lr2 = Train(w1, w2, lr1, lr2)
    m_train , b_train = np.polyfit(y_train, output_train, 1)    
    output_validation = Validation(w1, w2)
    m_validation , b_validation = np.polyfit(y_validation, output_validation, 1)
    
    Plot_results(output_train, 
                 output_validation, 
                 m_train, 
                 b_train,
                 m_validation,
                 b_validation)

    stop = timeit.default_timer()
    print('Epoch: {} \t, time: {:.3f}'.format(epoch+1, stop-start))
    print('MSE_train: {:.4f} \t, MSE_validation: {:.4f}'.format(MSE_train[epoch], MSE_validation[epoch]))
    print(m_train, b_train, m_validation, b_validation)
print('==> End of training ...')

# =============================================================================
# Test
# =============================================================================
def Test(w1, w2):
    sqr_err_epoch_test = []
    output_test = []
    
    for i in range(np.shape(x_test)[0]):
        x = np.reshape(x_test[i], (1,-1))
        # Feed-Forward
        # Layer 1
        net1 = np.matmul(x, w1)
        o1 = sigmoid(net1)
        # Layer 2
        net2 = np.matmul(o1, w2)
        o2 = net2

        output_test.append(o2[0])

        # Error
        err = y_test[i] - o2[0]
        sqr_err_epoch_test.append(err ** 2)

    mse_epoch_test = 0.5 * ((sum(sqr_err_epoch_test))/np.shape(x_test)[0])
    m_test , b_test = np.polyfit(y_test, output_test, 1)  
    
    # Plots
    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(8, 10)
    axs[0].plot(y_test, 'b')
    axs[0].plot(output_test,'r')
    axs[0].set_title('Output Test')

    axs[1].plot(y_test, output_test, 'b*')
    axs[1].plot(y_test, m_test*y_test+b_test,'r')
    axs[1].set_title('Regression Test')
    plt.show()
    plt.close(fig)
    return mse_epoch_test[0]
MSE_test = Test(w1, w2)
print('MSE_test: {:.4f}'.format(MSE_test))
