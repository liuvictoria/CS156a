import numpy as np
import random
import math

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

def generate_target_f():
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


def classify_point(random_point, slope, intercept):
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


def create_data(N, slope, intercept):
    """
    Creates N points for testing data using target f(X) = slope * X + intercept;
    Notes:
        Target function f is a line connecting two points in X space [-1, 1] x [-1, 1]
        Classification is based on whether points lie above or below f
        
    Inputs:
        N (int)
        slope (float), from generate_target_f
        intercept (float), from generate_target_f
        
    Outputs:
        X (ndarray), X.shape = (N, 3); x0 = 1
        Y (ndarray), Y.shsape = (N, )
    """
    # create matrix X, where x0 is always 1, to accomodate w0
    X = np.ones((N, 3))
    Y = np.zeros(N)
    for i in range(N):
        random_point = generate_point(-1, 1, 2)
        classification = classify_point(random_point, slope, intercept)
        X[i, 1:3] = random_point
        Y[i] = classification
        
    return (X, Y)


def perceptron_basic(X_training, Y_training):
    """
    Starts with the weight as a zero-vector.
    Changes the weight when it meets any misclassified point.
    Returns the new weight once all the points are correctly clasified.
    
    Inputs:
        X_training (ndarray), complete input features of all points
        Y_training (ndarray), complete training classification of all points
        
    Outputs:
        weight (ndarray), final hypothesis g(X) = Y for perceptron learning
        
    """
    feature_length = X_training.shape[1]
    weight = np.zeros(feature_length)

    while True:
        misclassified_point_count = 0
        for i in range(len(X_training)):
            if isit_misclassified_point(X_training[i], Y_training[i], weight) == True:
                weight = adjust_weight(X_training[i], Y_training[i], weight)
                misclassified_point_count += 1
        if misclassified_point_count == 0:
            break
            
    return weight

def perceptron(X_training, Y_training):
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
    feature_length = X_training.shape[1]
    weight = np.zeros(feature_length)
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

def perceptron_error_single_run(X_testing, Y_testing, weight):
    """
    g = (weight . x)
    Inputs:
        X_testing (ndarray), complete input features of testing set
        Y_testing (ndarray), complete accurate classification of testing set, from f(x)
        weight (ndarray), generated by perceptron() func as the first element of the tuple
    Outputs:
        pla_error_freq (float), P[f(x)!= g_pla(x)]
    """
    
    N = X_testing.shape[0]
    error_count = 0
    for i in range(N):
        if isit_misclassified_point(X_testing[i], Y_testing[i], weight):
            error_count += 1
            
    pla_error_freq = error_count / N
    return pla_error_freq

def experiment(N_training, N_testing, runs):
    """
    Inputs:
        N_training: (int), number of points for training
        N_testing: (int), number of points for testing
        runs: (int), number of runs
    """
   
    average_iteration = 0
    average_error = 0
    for i in range(runs):
            
        #generate random function f(X) for every run
        slope, intercept = generate_target_f()
        
        #create training data based on f(X)
        X_training, Y_training = create_data(N_training, slope, intercept)
        
        #get weights for best hypothesis g
        weight, iteration_count = perceptron(X_training, Y_training)
        average_iteration += iteration_count
        
        #create testing data based on f(X)
        X_testing, Y_testing = create_data(N_testing, slope, intercept)
        
        #use perceptron_error_single_run to get pla_error_freq
        pla_error_freq = perceptron_error_single_run(X_testing, Y_testing, weight)
        average_error += pla_error_freq
            
    #outside forloop
    average_iteration /= runs
    average_error /= runs
    
    return (average_iteration, average_error)

#problems 7-8
average_iteration, average_error = experiment(10, 10000, 1000)
print("For N = 10 \n")
print("Average number of iterations: ", average_iteration, "\n")
print("Average error between g and f: ", average_error, "\n\n\n")

#problems 9-10
average_iteration, average_error = experiment(100, 10000, 100)
print("For N = 100 \n")
print("Average number of iterations: ", average_iteration, "\n")
print("Average error between g and f: ", average_error, "\n\n\n")