
# coding: utf-8

# In[1]:

from preprocessed_mnist import load_dataset
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
print(X_train.shape, y_train.shape)
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.imshow(X_train[0], cmap="Greys");


# In[6]:

import tensorflow as tf
import numpy as np
import math
from tensorflow.python.framework import ops

X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
X_val_flatten = X_val.reshape(X_val.shape[0],-1).T
X_test_flatten = X_test.reshape(X_test.shape[0], -1).T

print(X_train_flatten.shape)
print(X_val_flatten.shape)
print(X_test_flatten.shape)


# In[7]:

# Take one vector of labes and the total number of classes C, and return the one hot encoding.
def one_hot_matrix(labels, C):
    """
    Create a matrix where the i-th row corresponds to the ith class number and the jth column corresponds to the jth training example
    So if example j had a lable i,  then entry (i,j) will be 1.

    Arguments:
    labels -- vector containing the lables
    C -- number of classes, the depth of one hot dimension
    """
    # Create a tf.constant equal to C(depth), name it 'C'
    C = tf.constant(C, name="C")
    
    # Use tr.one_hot, be careful with the axis
    one_hot_matrix = tf.one_hot(labels, depth=C, axis = 0)
    
    # creat the session
    sess = tf.Session()
    
    one_hot = sess.run(one_hot_matrix)
    
    sess.close()
    
    return one_hot


# encode y with one-hot
y_train_one_hot = one_hot_matrix(y_train, 10)
y_val_one_hot = one_hot_matrix(y_val, 10)
y_test_one_hot = one_hot_matrix(y_test, 10)

print(y_train_one_hot.shape)
print(y_val_one_hot.shape)
print(y_test_one_hot.shape)


# In[8]:

# Create placeholders for X and Y
def create_placeholders(n_x, n_y):
    """ 
    Create the placeholders for the tensorflow session
    Arguments: 
    n_x -- Scalar, size of an image vector
    n_y -- Scalar, number of classes 
    
    Return:
    X -- placeholder for the data input, of shape [n_x, None] and "float"
    Y -- placeholder for the input lables, of shape [n_y, none] and "float"
    """
    
    X = tf.placeholder(tf.float32, [n_x, None], name= "X")
    Y = tf.placeholder(tf.float32, [n_y, None], name = "Y")
    
    return X, Y


# In[23]:

# Initialize_Parameters in tensorflow
def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are?
    W1 : [50, 784]
    b1 : [50, 1]
    W2 : [10, 50]
    b2 : [10, 1]
    
    returns: 
    parameters -- a dictionary of tensors containing W1, b1, W2, b2
    """
    W1 =  tf.get_variable("W1", [50,784], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 =  tf.get_variable("b1", [50,1], initializer = tf.zeros_initializer())
    W2 =  tf.get_variable("W2", [10,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 =  tf.get_variable("b2", [10,1], initializer = tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2
                 }
    
    return parameters


# In[24]:

# Forward propagation
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: Linear ->RELU ->Linear ->SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape(input size, number of examples)
    parameter -- python dictionary containig parameters "W1", "b1", "W2", "b2"
    
    Returns:
    Z3 -- the output of the last linear unit
    """
    
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    
    return Z2
    


# In[25]:

# Comput cost
def compute_cost(Z2, Y):
    """ 
    computes the cost
    Arguments:
    Z2 -- output of forward propagation(output of the last Linear unit), of shape(10, number of examples)
    Y -- "TRue" label vector places holders, same shape as Z2
    
    Returns:
    cost -- tensor of cost function
    """
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z2)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    
    return cost


# In[13]:




# In[26]:

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- input target, of shape (10, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size : ]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size : ]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# In[ ]:




# In[ ]:




# In[ ]:




# In[27]:

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y =  create_placeholders(n_x, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z2 = forward_propagation(X, parameters)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z2, Y)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z2), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters


# In[ ]:

parameters = model(X_train_flatten, y_train_one_hot, X_val_flatten, y_val_one_hot)




