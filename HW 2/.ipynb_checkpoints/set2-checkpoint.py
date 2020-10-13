# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import random
import math

# +
#Problems 1-2

def flip_single_coin_n_times(n = 10):
    """
    binomial 1 means getting heads,
    binomial 0 means getting tails.
    We assume fair coin.
    We return the proportion of flips returning heads.
    """
    #flip one fair coin n times
    flips = np.random.binomial(1, .5, n)
    heads_fraction = np.sum(flips) / n
    return heads_fraction

def coin_experiment_single_run(n = 10, coin_count = 1000):
    #for c_rand
    random_index = np.random.randint(coin_count, size = 1)
    #list of size coin_count, for v of each coin
    heads_fraction_all = []
    
    for coin in range(coin_count):
        heads_fraction_single = flip_single_coin_n_times()
        heads_fraction_all.append(heads_fraction_single)
        
        #find v_0 and v_rand, if applicable
        if coin == 0:
            v_0 = heads_fraction_single
        #cannot use elif, since random_index might be 0
        if coin == random_index:
            v_rand = heads_fraction_single
    
    #find v_min
    heads_fraction_all = np.asarray(heads_fraction_all)
    v_min = np.min(heads_fraction_all)
    return (v_0, v_rand, v_min)

def coin_experiment_expected_values(n = 10, coin_count = 1000, runs = 100000):
    """
    run coin_experiment_single_run runs = 1000 times
    Outputs:
        v_0_expected (should be around .5)
        v_rand_expected (should be around .5)
        v_min_expected (should be a lot less than .5)
    """
    v_0_all = []
    v_rand_all = []
    v_min_all = []
    for run in range(runs):
        if run % 10000 == 0:
            print ("on run ", run)
        v_0, v_rand, v_min = coin_experiment_single_run(n = n, coin_count = coin_count)
        v_0_all.append(v_0)
        v_rand_all.append(v_rand)
        v_min_all.append(v_min)
    
    #convert to ndarray
    v_0_all = np.asarray(v_0_all)
    v_rand_all = np.asarray(v_rand_all)
    v_min_all = np.asarray(v_min_all)
    
    v_0_expected = np.sum(v_0_all) / runs
    v_rand_expected = np.sum(v_rand_all) / runs
    v_min_expected = np.sum(v_min_all) / runs
    
    return (v_0_expected, v_rand_expected, v_min_expected)

v_0_expected, v_rand_expected, v_min_expected = coin_experiment_expected_values()
print("Problems 1-2")
print ("expected v_0 is", v_0_expected, "\n")
print ("expected v_rand is", v_rand_expected, "\n")
print ("expected v_min is", v_min_expected, "\n")

#example output:
#Problems 1-2
#expected v_0 is 0.4994809999999999 

#expected v_rand is 0.4993659999999999 

#expected v_min is 0.03745199999999999 

# +
#problems 5-7
def generate_point(boundary1, boundary2, dimension):
    """
    Generate random two-dimensional point on 
    [boundary1, boundary 2] X [boundary1, boundary2] space.
    Returns ndarray of (x, y) point
    """
    random_point = np.zeros(dimension)
    for i in range(dimension):
        random_point[i] = np.random.uniform(boundary1, boundary2, 1)
    return random_point

def generate_target_f_linear():
    """
    Returns slope and intercept of line connecting two random points
    """
    point_1 = generate_point(-1, 1, 2)
    point_2 = generate_point(-1, 1, 2)
    
    # slope = (y2 - y1) / (x2 - x1)
    slope = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
    # intercept = y1 - slope * x1
    intercept = point_2[1] - slope * point_2[0]
    return(slope, intercept)


def classify_point_linear(random_point, slope, intercept):
    """
    Given random_point in (x, y) form and a slope and intercept,
    label the point
    +1 if it falls above the line
    -1 if it falls below the line
    """
    if random_point[1] > slope * random_point[0] + intercept:
        classification = 1
    else:
        classification = -1
    return classification


def create_data_linear(N, slope, intercept):
    """
    Creates N points for testing data using target f(X) = slope * X + intercept;
    Notes:
        Target function f is a line connecting two points in X space [-1, 1] x [-1, 1]
        Classification is based on whether points lie above or below f
        
    Inputs:
        N (int)
        slope (float), from generate_target_f_linear
        intercept (float), from generate_target_f_linear
        
    Outputs:
        X (ndarray), X.shape = (N, 3); x0 = 1
        Y (ndarray), Y.shsape = (N, )
    """
    # create matrix X, where x0 is always 1, to accomodate w0
    X = np.ones((N, 3))
    Y = np.zeros(N)
    for i in range(N):
        random_point = generate_point(-1, 1, 2)
        classification = classify_point_linear(random_point, slope, intercept)
        X[i, 1:3] = random_point
        Y[i] = classification
        
    return (X, Y)


def linear_regression_weights(X_training, Y_training):
    """
    Outputs:
        weights (ndarray), weights.shape = d + 1 (one-dimensional)
    """
    
    X_pseudo_inverse = np.linalg.pinv(X_training)
    weights = np.dot(X_pseudo_inverse, Y_training)
    return weights


def label_Y_g(X_training, weights):
    Y_g = np.sign(np.dot(X_training, weights))
    return Y_g
    

def error_linear(Y_f, Y_g):
    """
    Inputs:
        Y_f: (ndarray) Y_f.shape = (N, ); Y labels generated by target func
        Y_g: (ndarray) Y_g.shape = (N, ); Y labels generated by hypothesis
    """
    if Y_f.shape[0] != Y_g.shape[0]:
        raise RunTimeError('Y_f and Y_g are not the same length')
    correct_count = np.sum(Y_f == Y_g)
    error_percentage = (Y_f.shape[0] - correct_count) / Y_f.shape[0]
    return error_percentage


def perceptron(X_training, Y_training, weight = np.zeros(3)):
    """
    Starts with the weight as a zero-vector.
    In a given iteration, checks every single point for misclassification.
    At the end of the iteration, randomly chooses a misclassified point for weight adjustment.
    Returns the new weight once all the points are correctly classified.
    
    Inputs:
        X_training (ndarray), complete input features of all points
        Y_training (ndarray), complete training classification of all points
        
    Outputs:
        weight (ndarray), final hypothesis g(X) = Y for perceptron learning
        iteration_count (int), number of iterations for convergence between g and f
        
    """
    iteration_count = 0
    
    while True:
        misclassified_points = []
        for i in range(len(X_training)):
            if isit_misclassified_point(X_training[i], Y_training[i], weight) == True:
                misclassified_points.append(i)
        if len(misclassified_points) == 0:
            break
            
        random_i = random.choice(misclassified_points)
        weight = adjust_weight(X_training[random_i], Y_training[random_i], weight)
        iteration_count += 1
        
    return (weight, iteration_count)

def adjust_weight(x_misclassified, y_misclassified, weight):
    """
    Given a single misclassified point and the current weight vector,
    adjust the weight to accomodate our misclassifed point.
    
    Inputs:
        x_misclassified (ndarray)
        y_misclassified (+/- 1)
        weight (ndarray)
        
    Outputs:
        misclassified (boolean)
    """
    adjusted_weight = weight + np.dot(y_misclassified, x_misclassified)
    return adjusted_weight


def isit_misclassified_point(x, y, weight):
    """
    Given a single point (i.e. vector x and label y)
    and a weight that we are currently training or testing, it will determine whether the
    current hypothesis weight correctly or incorrectly classifies the point.
    Tests hypothesis g(x) = sign (weight . x) == y for classification
    This is primarily used for pla, since there is a more computationally friendly
    way for linear regression.
    
    Inputs:
        x (ndarray)
        y (+/- 1)
        weight (must be ndarray)
        
    Outputs:
        misclassified (boolean)
    
    """
    if np.sign(np.dot(weight.T, x)) != y:
        miclassified = True
    else:
        miclassified = False
    return miclassified

def experiment_5(N_training, N_testing, runs = 1000):
    for run in range(runs):
        #generate linear target function on [-1, 1] x [-1, 1]
        slope, intercept = generate_target_f_linear()
        
        #generate training, and then one-step learning
        X_training, Y_training = create_data_linear(N_training, slope, intercept)
        weights = linear_regression_weights(X_training, Y_training)
        
        #how does g label Y_training?
        Y_training_g = label_Y_g(X_training, weights)
        
        #calculate in-sample error
        e_in = error_linear(Y_training, Y_training_g)
        
        #generate testing data
        X_testing, Y_testing = create_data_linear(N_training, slope, intercept)
        
        #how does g label Y_testing?
        Y_testing_g = label_Y_g(X_testing, weights)
        
        #calculate out-of-sample error
        e_out = error_linear(Y_testing, Y_testing_g)
        
    return (e_in, e_out)

def experiment_7(N_training, runs = 1000):
    iterations_tally = 0
    for run in range(runs):
        #generate linear target function on [-1, 1] x [-1, 1]
        slope, intercept = generate_target_f_linear()
        
        #generate training, and then one-step learning
        X_training, Y_training = create_data_linear(N_training, slope, intercept)
        weights = linear_regression_weights(X_training, Y_training)
        _, iterations = perceptron(X_training, Y_training, weight = weights)
        iterations_tally += iterations
    average_iterations = iterations_tally / runs
    return average_iterations

#homework answers

print("Problems 5/6")
error_in, error_out = experiment_5(100, 1000, runs = 1000)
print("In sample error is", error_in)
print("Out of sample error is", error_out)
print("\n")
print("Problem 7")
print("Iterations until convergence:", experiment_7(10, runs = 1000))

#example output:
#Problems 5/6
#In sample error is 0.03
#Out of sample error is 0.01

#Problem 7
#Iterations until convergence: 3.78

# +
#problems 8-10

def my_target_function(x_1, x_2):
    return (x_1**2 + x_2**2 - 0.6)


def classify_point(random_point, my_target_function):
    """
    Given random_point in (x_1, x_2) form and a target function,
    label the point
    +1 if my_target_function(x_1, x_1) > 0
    -1 if my_target_function(x_1, x_1) < 0
    Inputs:
        random_point (x_1, x_2)
        my_target_function(*random_point) represents a graph
        Note that there are no *kwargs because random_point provides the input
    """
    if np.sign(my_target_function(*random_point)) > 0:
        classification = 1
    else:
        classification = -1
    return classification


def create_data(N, my_target_function):
    """
    Creates N points for testing data using my_target_function that has been defined elsewhere
    Notes:
        Target function f is a function on X space [-1, 1] x [-1, 1]
        Classification y is based on the sign of f(x)
        
    Inputs:
        N (int)
        my_target_function(*kwargs) represents a graph
        
    Outputs:
        X (ndarray), X.shape = (N, 3); x0 = 1
        Y (ndarray), Y.shsape = (N, )
    """
    
    # create matrix X, where x0 is always 1, to accomodate w0
    X = np.ones((N, 3))
    Y = np.zeros(N)
    for i in range(N):
        random_point = generate_point(-1, 1, 2)
        classification = classify_point(random_point, my_target_function)
        X[i, 1:3] = random_point
        Y[i] = classification
        
    return (X, Y)


def create_noise(Y_training, percent_noise):
    #calculate number of classifications to flip
    N = Y_training.shape[0]
    flip_count = int(N * percent_noise)
    
    #determine random indices of Y_training to flip
    flip_inds = np.random.choice(range(N), size = flip_count, replace = False)
    
    #flip using fancy numpy indexing
    Y_training[flip_inds] = -Y_training[flip_inds]
    return Y_training


def linear_regression_weights(X_training, Y_training):
    """
    Outputs:
        weights (ndarray), weights.shape = d + 1 (one-dimensional)
    """
    
    X_pseudo_inverse = np.linalg.pinv(X_training)
    weights = np.dot(X_pseudo_inverse, Y_training)
    return weights


def X_to_Z_matrix(X_training, non_linear_transform):
    """
    Inputs:
        X_training (ndarray), X_training.shape = (N, d+1);
            has feature vector (1, x_1, x_2)
    
    Outputs:
        Z_training (ndarray), Z_training.shape = (N, 6)
    """
    N = X_training.shape[0]
    Z_training = np.ones([N, 6])
    
    #transform the matrix, row by row using non_linear_transform
    for i in range(N):
        Z_training[i, :] = non_linear_transform(X_training[i])
        
    return Z_training


def non_linear_transform(x):
    """
    Inputs:
        x: (ndarray), x.shape = d + 1 (one dimensional)
    Outputs:
        z: (ndarray), z.shape = 6 (one dimensional)
        
    Nonlinear feature vector:
        (1, x_1, x_2, x_1 * x_2, x_1**2, x_2**2)
    
    """
    _, x_1, x_2 = x
    z = [1, x_1, x_2, x_1 * x_2, x_1**2, x_2**2]
    z = np.asarray(z)
    return z


def experiment_8(N = 1000, runs = 1000):
    error_in_tally = 0
    for run in range(runs):
        #create training data
        X_training, Y_training = create_data(N, my_target_function)

        #add 10% noise
        Y_training = create_noise(Y_training, .1)

        #linear regression
        weights = linear_regression_weights(X_training, Y_training)

        #how would our linear regression label Y_training?
        Y_training_g = label_Y_g(X_training, weights)

        #calculate in sample error
        error_in = error_linear(Y_training, Y_training_g)
        
        #add to tally
        error_in_tally += error_in
    error_in_avg = error_in_tally / runs
    return error_in_avg

def experiment_9(N_training = 1000, N_testing = 1000, runs = 1000):
    error_out_tally = 0
    weights_tally = []
    for run in range(runs):
        #create training data
        X_training, Y_training = create_data(N_training, my_target_function)

        #add 10% noise
        Y_training = create_noise(Y_training, .1)
        
        #nonlinear transformation to Z-space
        Z_training = X_to_Z_matrix(X_training, non_linear_transform)

        #linear regression and adding to tally
        weights = linear_regression_weights(Z_training, Y_training)
        weights_tally.append(weights)
        
        #generate testing data
        X_testing, Y_testing = create_data(N_testing, my_target_function)
        
        #add 10% noise to testing data
        Y_testing = create_noise(Y_testing, .1)
        
        #nonlinear transformation to Z-space
        Z_testing = X_to_Z_matrix(X_testing, non_linear_transform)
        
        #how would our linear regression label Y_training?
        Y_testing_g = label_Y_g(Z_testing, weights)

        #calculate out of sample error
        error_out = error_linear(Y_testing, Y_testing_g)
        
        #add to tally
        error_out_tally += error_out
        
    #getting averages    
    error_out_avg = error_out_tally / runs
    
    average_weights = []
    weights_tally = np.asarray(weights_tally)
    for i in range(6):
        mean_weight = np.mean(weights_tally.T[i])
        average_weights.append(mean_weight)
    average_weights = np.asarray(average_weights)
    
    return error_out_avg, average_weights

print("Problem 8")
print("Average in-sample error is", experiment_8())

#sample output
#Problem 8
#Average in-sample error is 0.5073309999999999


print("Problems 9/10")
error_out_avg, c = experiment_9(runs = 1000)
regression = f"{c[0]} + {c[1]}x_1 + {c[2]}x_2 + {c[3]}x_1*x_2 + {c[4]}x_1**2 + {c[5]}x_2**2"
print ("Regression equation \n", regression)
print ("Average out of sample error", error_out_avg)

#sample output
#Problems 9/10
#Regression equation 
#-0.9928129029169119 + -0.0009431377749000227x_1 + 0.001174531635834422x_2 + -0.0027519898287829176x_1*x_2 + 1.5549160903595107x_1**2 + 1.5611160733958145x_2**2
#Average out of sample error 0.12664100000000014
# -




