import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import skimage 
from skimage import transform 
from PIL import Image
from scipy import ndimage
import cv2

from Linear_forward import linear_activation_forward
from Linear_backward import linear_activation_backward
from Compute_Cost import compute_cost

def load_data():
    train_dataset = h5py.File("C:\\Users\\Ege\\Desktop\\NeuralNetwork\\DataSets\\train_all_64.hdf5", "r")
    train_set_x_orig = np.array(train_dataset["train_img"][:])                  # your train set features
    train_set_y_orig = np.array(train_dataset["train_labels"][:])               # your train set labels

    test_dataset = h5py.File("C:\\Users\\Ege\\Desktop\\NeuralNetwork\\DataSets\\test_plastic_glass_metal_64.hdf5", "r")
    test_set_x_orig = np.array(test_dataset["test_img"][:])                     # test set features
    test_set_y_orig = np.array(test_dataset["test_labels"][:])  

    classes = np.array(test_dataset["list_classes"][:])                         # the list of classes
    
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters  

def initialize_parameters_deep(layer_dims):

    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

def Model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)

    # Implement LINEAR -> SOFTMAX. Add "cache" to the "caches" list.
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "softmax")
    caches.append(cache)
    
    assert(AL.shape == (3,X.shape[1]))
            
    return AL, caches

def Model_backward(AL, Y, caches):

    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dZ = AL - Y
    
    # Lth layer (SOFTMAX -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dZ, current_cache, activation = "softmax")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def predict(X, y, parameters):
    
    m = X.shape[1]
    n = len(parameters)  # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = Model_forward(X, parameters)
    
    # convert probas to 0/1/2 predictions
    p = np.argmax(probas,axis=0)

    #print results
    print ("\nPredictions: " + str(p))
    print ("\nTrue labels: " + str(y))
    print ("\nAccuracy: "  + str(np.sum((p == y)/m)))
        
    return p

def Layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    
    np.random.seed(1)
    costs = []                         # keep track of cost
    
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SOFTMAX.
        AL, caches = Model_forward(X,parameters)

        # Compute cost.
        cost = compute_cost(Y,AL)
    
        # Backward propagation.
        grads = Model_backward(AL,Y,caches)
 
        # Update parameters.
        parameters = update_parameters(parameters,grads,learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

if __name__ == "__main__":
    
    ######################################################################################################
    #                              DataSets Take With load_data() Function                               #
    ######################################################################################################

    train_x_orig,train_y,test,test_y,classes = load_data()

    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T
    train_x = train_x_flatten/255.    
    test_x_flatten = test.reshape(test.shape[0],-1).T
    test_x = test_x_flatten/255.

    digits = 3
    train_y = np.reshape(train_y,(1,600))
    examples = train_y.shape[1]
    Y_new = np.eye(digits)[train_y.astype('int32')]
    Y_new = Y_new.T.reshape(digits, examples)

    ######################################################################################################
    #                                      Info About Datasets                                           #
    ######################################################################################################

    print("\n")
    print("Number of Traing Examples  : " + str(train_x_orig.shape[0]))
    print("Number of Testing Examples : " + str(test.shape[0]))
    print("Each Image is of Size      : (" + str(64) + ", " + str(64) + " , 3)")
    print("traing_x_orig shape        : " + str(train_x_orig.shape))
    print("train_y shape              : " + str(train_y.shape))
    print("test shape                 : " + str(test.shape))
    print("test_y shape               : " + str(test_y.shape))

    print("\n")
    print("Started Iteration For Training : ")
    
    ######################################################################################################
    #                               Model Start With Layer_model Function                                #
    ######################################################################################################

    layers_dims = [12288,64,20,7,5,3]
    print("Number of Layers(64,20,7,5,3) S: " + str(len(layers_dims)-1))
    print("\n") 
    parameters = Layer_model(train_x,Y_new,layers_dims,num_iterations=2000, print_cost=True)

    ######################################################################################################
    #                                             Predict                                                #
    ######################################################################################################
    
    print("\n")
    print("Plastic : 0")
    print("Glass   : 1")
    print("Metal   : 2")
    print("\n")

    image = test[22]
    img_label = test_y[0][22]

    file = 'test.jpg'
    cv2.imwrite(file,image)

    img = cv2.imread(file,0)
    resized = cv2.resize(img,(400,400),interpolation = cv2.INTER_AREA)
    cv2.imshow('Image',resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = skimage.transform.resize(image,(64,64)).reshape((64*64*3,1))
    img = img/255.

    predict(img,img_label,parameters)
    final = predict(test_x,test_y,parameters)
