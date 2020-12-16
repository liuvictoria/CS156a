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

# +
import numpy as np

#create random number generator
rg = np.random.default_rng()

# +
#problems 5-7

def nonlinear_error(u, v):
    """
    as described in the problem
    """
    return (u * np.exp(v) - 2 * v * np.exp(-u)) ** 2

def dE_du(u, v):
    """
    the direction of steepest increase in the u direction
    """
    return 2 * (np.exp(v) + 2 * v * np.exp(-u)) * (u * np.exp(v) - 2 * v * np.exp(-u))

def dE_dv(u, v):
    """
    the direction of steepest increase in the v direction
    """
    return 2 * (u * np.exp(v) - 2 * np.exp(-u)) * (u * np.exp(v) - 2 * v * np.exp(-u))

def gradient_descent(u = 1, v = 1, eta = .1, error_thres = 10**(-14)):
    """
    gradient descent with the given parameters
    start at u = 1 and v = 1
    outputs u, v, and the iterations to get below a certain error threshold
    """
    iteration = 0
    error = nonlinear_error(u, v)
    while error > error_thres:
        du = dE_du(u, v)
        dv = dE_dv(u, v)
        u -= eta * du
        v -= eta * dv
        error = nonlinear_error(u, v)
        iteration += 1
    return (u, v, iteration)


#problem 7
def coordinate_descent(u = 1, v = 1, eta = .1, stop = 15):
    """
    rather than doing gradient descent, we do coordinate descent,
    where we go in the steepest direction of one coordinate,
    use the new location to calculate the other coordinate's steepest decline, etc.
    """
    iteration = 0
    error = nonlinear_error(u, v)
    while iteration < stop:
        du = dE_du(u, v)
        u -= eta * du
        dv = dE_dv(u, v)
        v -= eta * dv
        error = nonlinear_error(u, v)
        iteration += 1
    return (error)

u, v, iteration = gradient_descent()
print(f'u: {u}')
print(f'v: {v}')
print(f'iterations: {iteration}')

print()

error = coordinate_descent()
print(f'error for coordinate descent: {error}')

#sample output
# u: 0.04473629039778207
# v: 0.023958714099141746
# iterations: 10

# error for coordinate descent: 0.13981379199615315

# +
#problems 8-9
#we co-opt / modify some of the code we wrote for pset1/2:

def generate_point(boundary1, boundary2, dimension):
    """
    Generate random dimension-dimensional point on 
    [boundary1, boundary 2] X [boundary1, boundary2] X etc. space.
    Returns ndarray of (x1, x2) point
    """
    #set random number generator
    rg = np.random.default_rng()
    
    random_point = np.zeros(dimension)
    for i in range(dimension):
        random_point[i] = rg.uniform(low = boundary1, high = boundary2, size = 1)
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



def cross_entropy_error_gradient(x, y, weights):
    """
    cross entropy gradient, slide 23 of lecture 9
    for a single point at a time
    """
    return - y * x/(1 + np.exp(y * np.dot(x, weights)))



def update_weight(x, y, weights, eta):
    """
    update weight according to a learning rate and cross_entropy_error
    """
    weights -= eta * cross_entropy_error_gradient(x, y, weights)
    return weights



def train_weights(slope, intercept, eta = .01, training_size = 100, stop = .01):
    """
    Train with error measure as logistic regression's cross entropy error; 
    Inputs:
        eta (learning rate)
        training_size (size of training set)
        stop (when subsequent weight adjustments are less than stop, return)
    Outputs:
        weights_2 (final weights based on given parameters)
        epoch (how many epochs to get to weights_2)
    """
    
    #generate training data
    X_training, Y_training = create_data_linear(training_size, slope, intercept)
    
    #initialize epoch counter and weights; keep two weights to allow comparisons
    epoch = 0
    weights_1 = np.ones(3)
    weights_2 = np.zeros(3)
    
    #learn using sgd and logistic regression
    while np.linalg.norm(weights_2 - weights_1) >= 0.01:
        weights_1 = weights_2.copy()

        #randomize data for presentation to stochastic g.d. with a random index
        random_index = rg.choice(np.arange(training_size), size = training_size, replace = False)
        
        for index in random_index:
            x_point, y_point = X_training[index], Y_training[index]
            weights_2 = update_weight(x_point, y_point, weights_2, eta)

        epoch += 1
    
    return (weights_2, epoch)



def test_weights(slope, intercept, testing_size = 100, eta = .01, training_size = 100, stop = .01):
    """
    determine E_out for weights determined by train_weights() function
    testing on a separate set of data
    Outputs:
        cross_entropy_error_out (E_out)
        epoch (epochs taken to train the model)
    """
    
    #generate testing data
    X_testing, Y_testing = create_data_linear(testing_size, slope, intercept)
    
    #what does the trained model say?
    weights, epoch= train_weights(
        slope, intercept, eta = eta, training_size = training_size, stop = stop
        )
    
    #calcalate test sample error
    cross_entropy_error_out = 0
    for i in range(testing_size):
        #slide 16, lecture 9
        cross_entropy_error_out += np.log(1 + np.exp(-Y_testing[i] * np.dot(weights, X_testing[i])))
    
    return (cross_entropy_error_out / testing_size, epoch)


def experiment(runs = 1000, testing_size = 100, eta = .01, training_size = 100, stop = .01):
    """
    Do problems 8/9 using parameters given, and helper functions above
    """
    
    slope, intercept = generate_target_f_linear()
    
    #initialize counters to sum up and average over the runs
    epoch_counter = 0
    e_out_counter = 0
    
    #
    for run in range(runs):
        if (run + 1) % 100 == 0:
            print ('run', run + 1)
        cross_entropy_error_out, epoch = test_weights(
            slope, intercept, testing_size = testing_size, eta = eta, training_size = training_size, stop = stop
            )
        e_out_counter += cross_entropy_error_out
        epoch_counter += epoch
        
    return e_out_counter / run, epoch_counter / run


e_out, epoch = experiment()

print(f"The average E_out is {e_out}")
print(f"The average epoch to get there is {epoch}")

#sample output
# run 100
# run 200
# run 300
# run 400
# run 500
# run 600
# run 700
# run 800
# run 900
# run 1000
# The average E_out is 0.11080479417685717
# The average epoch to get there is 374.83983983983984
