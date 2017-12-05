# Modules used
from multiprocessing import Queue

# From Library by authors
from lib.solver import Solver
from lib.data_io import category_model_id_pair
from lib.data_process import kill_processes, make_data_processes

# My reimplementation
from my_3DR2N2.my_res_gru_net import My_ResidualGRUNet

# Define globally accessible queues, will be used for clean exit when force
train_queue, validation_queue, train_processes, val_processes = None, None, None, None

# Clean up in case of unexpected quit
def cleanup_handle(func):
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            print('Wait until the dataprocesses to end')
            kill_processes(train_queue, train_processes)
            kill_processes(validation_queue, val_processes)
            raise

    return func_wrapper


# Train the network
@cleanup_handle
def train_net():

    # Set up the model and the solver
    my_net = My_ResidualGRUNet()

    # Generate the solver
    solver = Solver(my_net)

    # Load the global variables
    global train_queue, validation_queue, train_processes, val_processes

    # Initialize the queues
    train_queue = Queue(15)  # maximum number of minibatches that can be put in a data queue)
    validation_queue = Queue(15)

    # Train on 80 percent of the data
    train_dataset_portion = [0, 0.8]

    # Validate on 20 percent of the data
    test_dataset_portion = [0.8, 1]

    # Establish the training procesesses
    train_processes = make_data_processes(
        train_queue,
        category_model_id_pair(dataset_portion=train_dataset_portion),
        1,
        repeat=True)

    # Establish the validation procesesses
    val_processes = make_data_processes(
        validation_queue,
        category_model_id_pair(dataset_portion=test_dataset_portion),
        1,
        repeat=True,
        train=False)

    # Train the network
    solver.train(train_queue, validation_queue)

    # Cleanup the processes and the queue.
    kill_processes(train_queue, train_processes)
    kill_processes(validation_queue, val_processes)


train_net()
