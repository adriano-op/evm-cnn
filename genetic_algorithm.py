import models

import pickle
import numpy as np
from deap import algorithms, base, tools, creator
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import EarlyStopping
from keras.utils import get_custom_objects
from keras.backend import sigmoid
from keras.models import Model
import array, random

# np.random.seed()

class SwishActivation(Activation):
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

max_conv_layers = 0
max_dense_layers = 4 # precisa de pelo menos um para o softmax

filter_range_max = 512
kernel_range_max = 7
max_dense_nodes = 512

# fixado para A AD100
input_shape = (1920, 64) # input_shape = (window_size, num_channels)
n_classes = 109

def decode(genome, verbose=False):
    batch_normalization=True
    dropout=True
    max_pooling=True
    optimizers=None 
    activations=None

    optimizer = [
        'adam',
        'rmsprop',
        'adagrad',
        'adadelta'
    ]

    activation =  [
        'relu',
        'sigmoid',
        swish_act,
    ]

    convolutional_layer_shape = [
        "active",
        "num filters",
        "kernel_size",
        "batch normalization",
        "activation",
        "dropout",
        "max pooling activation",
        "max pooling size",
    ]

    dense_layer_shape = [
        "active",
        "num nodes",
        "batch normalization",
        "activation",
        "dropout",
    ]

    convolution_layers = max_conv_layers
    convolution_layer_size = len(convolutional_layer_shape)
    dense_layers = max_dense_layers # excluindo a softmax layer
    dense_layer_size = len(dense_layer_shape)

    model = models.create_model_mixed(window_size, num_channels, num_classes)
    x = model.output

    offset = 0

    # mapear entre um range para outro (fazer de forma mais eficiente!)
    def map_range(value, leftMin, leftMax, rightMin, rightMax):
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)

        # Convert the 0-1 range into a value in the right range.
        return rightMin + (valueScaled * rightSpan)

    for i in range(dense_layers):
        if round(genome[offset])==1:
            dense = None
            dense = Dense(round(map_range(genome[offset + 1],0,1,4,max_dense_nodes)))
            x = Dense(round(map_range(genome[offset + 1],0,1,4,max_dense_nodes))) (x)
            if round(genome[offset + 2]) == 1:
                x = BatchNormalization()(x)
        
            x = Activation(activation[round(map_range(genome[offset + 3],0,1,0,len(activation)-1))])(x)
            x = Dropout(float(map_range(genome[offset + 4], 0, 1, 0, 0.7)))(x)

            if verbose==True:
                print('\n Dense%d' % i)
                print('Max Nodes = %d' % round(map_range(genome[offset + 1],0,1,4,max_dense_nodes)))
                if round(genome[offset + 2]) == 1:
                    print('Batch Norm')
                print('Activation=%s' % activation[int(round(genome[offset + 3]))])
                print('Dropout=%f' % float(map_range(genome[offset + 5], 0, 1, 0, 0.7)))
            
        offset += dense_layer_size

    predictions = Dense(n_classes, activation="softmax")(x)
    model = Model(inputs = model.input, outputs = predictions)

    model.compile(loss='categorical_crossentropy',
        optimizer=optimizer[round(map_range(genome[offset],0,1,0,len(activation)-1))],
        metrics=["accuracy"])

    if verbose==True:
        print('\n Optimizer: %s \n' % optimizer[round(map_range(genome[offset],0,1,0,len(activation)-1))])

    return model

def evaluate_individual(genome): 
    n_epochs = 8
    model = decode(genome, True)
    loss, accuracy, num_parameters = None, None, None

    fit_params = {
        'x': x_train,
        'y': y_train,
        'validation_split': 0.2,
        'epochs': n_epochs,
        'verbose': 1,
        'callbacks': [
            EarlyStopping(monitor='val_loss', patience=2, verbose=1)
        ]
    }

    fit_params['validation_data'] = (x_val, y_val)

    model.fit(**fit_params)
    (loss, accuracy) = model.evaluate(x_val, y_val, verbose=0)
    num_parameters = model.count_params()

    # return loss
    return accuracy

def prepare_toolbox(problem_instance, number_of_variables, bounds_low, bounds_up):
    
    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    
    toolbox = base.Toolbox()
    
    toolbox.register('evaluate', problem_instance)
    toolbox.register('select', tools.selRoulette)
    
    toolbox.register("attr_float", uniform, bounds_low, bounds_up, number_of_variables)
    toolbox.register("individual1", tools.initIterate, creator.Individual1, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual1)
    
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                     low=bounds_low, up=bounds_up, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, 
                     low=bounds_low, up=bounds_up, eta=20.0, 
                     indpb=1.0/number_of_variables)

    # default
    toolbox.pop_size = 10   # population size
    toolbox.max_gen = 5     # max number of iteration
    toolbox.mut_prob = 1/number_of_variables
    toolbox.cross_prob = 0.3
    
    return toolbox

def ga(toolbox, tools, pop_size, num_generations, recover_last_run=None, checkpoint=None):
    if recover_last_run and checkpoint:
        print("\nRetomando ultima execucao.. ]")
        # A file name has been given, then load the data from the file
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
    else:
        print("\nIniciando nova evolucao ]")
        # Start a new evolution
        population = toolbox.population(n=pop_size)
        start_gen = 0
        halloffame = tools.HallOfFame(maxsize=3)
        logbook = tools.Logbook()

    NGEN = num_generations
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    hof = tools.HallOfFame(3)
    
    for gen in range(start_gen, NGEN):
        print('\n **** Geracao %d  ****' % gen )
        population = algorithms.varAnd(population, toolbox, cxpb=0.4, mutpb=0.1)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

        population = toolbox.select(population, len(population))

        #if gen % FREQ == 0: # SALVA TODAS GERAÇÕES
        # Fill the dictionary using the dict(key=value[, ...]) constructor
        cp = dict(population=population, generation=gen, halloffame=halloffame,
                  logbook=logbook, rndstate=random.getstate())

        if checkpoint:
            with open(checkpoint, "wb") as cp_file:
                pickle.dump(cp, cp_file)

    # Print top N solutions 
    # best_individuals = tools.selBest(halloffame, k = 3) # if evaluate_individual returns loss
    best_individuals = tools.selWorst(halloffame, k = 3)  # if evaluate_individual returns accuracy
    
    print("\n\n ******* Best solution is: *******\n")
    for bi in best_individuals:
        decode(bi, True)
      
    print("\n")
    print("\n")
    return best_individuals

def genetic_run():

    population_size = 5     # num of solutions in the population
    num_generations = 8     # num of time we generate new population

    creator.create("FitnessMax1", base.Fitness, weights=(-1.0,) * 1)
    creator.create("Individual1", array.array, typecode='d', fitness=creator.FitnessMax1)

    number_of_variables = max_dense_layers*5 + 1 # convlayers, GAPlayer, denselayers, optimizer

    bounds_low, bounds_up = 0, 1 # valores sao remapeados em decode

    toolbox = prepare_toolbox(evaluate_individual, 
                              number_of_variables,
                              bounds_low, bounds_up)

    # chama o metodo genetico
    best_individuals = ga(toolbox, tools, population_size, num_generations)
    return best_individuals

best_individuals = genetic_run()

i = 1
for ind in best_individuals:
    print(f'accuracy do individuo #{i}: {ind.fitness.values}')
    i += 1

model = decode(best_individuals[0], True)
model.summary()