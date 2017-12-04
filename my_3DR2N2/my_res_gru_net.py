# Modules used
import numpy as np
import theano
import theano.tensor as tensor
import datetime as dt


# Taken from original repository. All of the layers were made by authors
from lib.layers import TensorProductLayer, ConvLayer, PoolLayer, Unpool3DLayer, \
    LeakyReLU, SoftmaxWithLoss3D, Conv3DLayer, InputLayer, FlattenLayer, \
    FCConv3DLayer, TanhLayer, SigmoidLayer, ComplementLayer, AddLayer, \
    EltwiseMultiplyLayer, get_trainable_params

# Initialize the tensor object
tensor5 = tensor.TensorType(theano.config.floatX, (False,) * 5)

# The Residual Gated Recurrent Unit Network my reimplementation
class My_ResidualGRUNet():

    # Initialize network parameters
    def __init__(self, random_seed=dt.datetime.now().microsecond, batch=36):

        # Initialize the random number generator
        self.rng = np.random.RandomState(random_seed)

        # Set the batch size
        self.batch_size = batch

        # Images should have a width and hieght of 127 x 127
        self.img_width = 127
        self.img_height = 127

        # The 3D Convolutional LSTM will be of 4 x 4 x 4 dimenstions
        self.n_gru_vox = 4

        # The input and output values are a 5D tensor object
        self.x = tensor5()
        self.y = tensor5()

        # list of activation functions
        self.activations = []

        # Final loss of network
        self.loss = []

        # Predicted output
        self.output = []

        # Error on predicted output when training
        self.error = []

        # Weights of all the layers
        self.params = []

        # Create the network structure
        self.network_definition()

    def network_definition(self):

        # Depth of the convolutional layers. VGG Style
        cnn_filters = [96, 128, 256, 256, 256, 256]

        # One fully connected layer for a 1024 feature vector
        fully_connecter_filter = [1024]

        # Shape of input layers. Used by encoder and GRU
        input_shape = (self.batch_size, 3, self.img_width, self.img_height)

        ######### Encoder ##########

        # Input Layer
        x = InputLayer(input_shape)

        ## First set of convolutional layers ##
        conv1a = ConvLayer(x, (cnn_filters[0], 7, 7)) # 96 x 7 x 7
        conv1b = ConvLayer(conv1a, (cnn_filters[0], 3, 3)) # 96 x 3 x 3
        pool1 = PoolLayer(conv1b) # Max Pooling

        ## Second set of convolutional layers ##
        conv2a = ConvLayer(pool1, (cnn_filters[1], 3, 3)) # 128 x 3 x 3
        conv2b = ConvLayer(conv2a, (cnn_filters[1], 3, 3)) # 128 x 3 x 3
        conv2c = ConvLayer(conv2b, (cnn_filters[1], 1, 1)) # 128 x 1 x 1
        pool2 = PoolLayer(conv2c) # Max Pooling

        ## Third set of convolutional layers ##
        conv3a = ConvLayer(pool2, (cnn_filters[2], 3, 3)) # 256 x 3 x 3
        conv3b = ConvLayer(conv3a, (cnn_filters[2], 3, 3)) # 256 x 3 x 3
        conv3c = ConvLayer(pool2, (cnn_filters[2], 1, 1)) # 256 x 1 x 1
        pool3 = PoolLayer(conv3b) # Max Pooling

        ## Fourth set of convolutional layers ##
        conv4a = ConvLayer(pool3, (cnn_filters[3], 3, 3)) # 256 x 3 x 3
        conv4b = ConvLayer(conv4a, (cnn_filters[3], 3, 3)) # 256 x 3 x 3
        pool4 = PoolLayer(conv4b) # Max Pooling

        ## Fifth set of convolutional layers ##
        conv5a = ConvLayer(pool4, (cnn_filters[4], 3, 3)) # 256 x 3 x 3
        conv5b = ConvLayer(conv5a, (cnn_filters[4], 3, 3)) # 256 x 3 x 3
        conv5c = ConvLayer(pool4, (cnn_filters[4], 1, 1)) # 256 x 1 x 1
        pool5 = PoolLayer(conv5b) # Max pooling

        ## Sixth set of convolutional layers ##
        conv6a = ConvLayer(pool5, (cnn_filters[5], 3, 3)) # 256 x 3 x 3
        conv6b = ConvLayer(conv6a, (cnn_filters[5], 3, 3)) # 256 x 3 x 3
        pool6 = PoolLayer(conv6b)

        # Flatten layer
        flat6 = FlattenLayer(pool6)

        # Fully Connected layer
        fc7 = TensorProductLayer(flat6, 1024) # 1024 feature vector

        ########## End Encoder ############


        ########## Gated Recurrent Unit ############

        # Filter size of layers within the unit
        gru_filters = [96, 128, 256, 256, 256, 256]

        # The 3D Convolutional LSTM has a grid structure of 4 x 4 x 4. 128 for first layer of decoder
        s_shape = (self.batch_size, self.n_gru_vox, gru_filters[1], self.n_gru_vox, self.n_gru_vox)

        # Initialize the first previous state to nothing
        prev_s = InputLayer(s_shape) # h(t-1)

        # 3 x 3 x 3 Convolution of hidden states of self and neighbors
        # Wfx T(xt) (+) Uf * h(t-1) + bf
        update_layer = FCConv3DLayer(prev_s, fc7, (gru_filters[1], gru_filters[1], 3, 3, 3)) # 128 x 3 x 3 x 3

        # Wix T(xt) (+) Ui * h(t-1) + bi
        reset_layer = FCConv3DLayer(prev_s, fc7, (gru_filters[1], gru_filters[1], 3, 3, 3)) # 128 x 3 x 3 x 3

        # Sigmoid (Wix T(xt) (+) Ui * h(t-1) + bi)
        reset_gate = SigmoidLayer(reset_layer)

        # rt (.) h(t-1)
        rs = EltwiseMultiplyLayer(reset_gate, prev_s) # Used  for h(t)

        # Wh T(xt) (+) Uh * (rt (.) h(t-1) + bh
        hidden_state_layer = FCConv3DLayer(rs, fc7, (gru_filters[1], gru_filters[1], 3, 3, 3)) # 128 x 3 x 3 x 3

        # Recurrence unit
        def recurrence(x_curr, prev_s_tensor, prev_in_gate_tensor):

            # Input layer
            input_ = InputLayer(input_shape, x_curr)

            # GRU network same parameters as encoder
            # Conv -> leakyReLU -> Conv -> LeakyReLU -> MaxPooling
            conv1a_ = ConvLayer(input_, (gru_filters[0], 7, 7), params=conv1a.params) # 96 x 7 x 7
            rect1a_ = LeakyReLU(conv1a_)
            conv1b_ = ConvLayer(rect1a_, (gru_filters[0], 3, 3), params=conv1b.params) # 96 x 3 x 3
            rect1_ = LeakyReLU(conv1b_)
            pool1_ = PoolLayer(rect1_)


            # Residual                               |=> -----------------=V
            # Conv -> leakyReLU -> Conv -> LeakyReLU -> Conv -> LeakyReLU -> MaxPooling
            conv2a_ = ConvLayer(pool1_, (gru_filters[1], 3, 3), params=conv2a.params) # 128 x 3 x 3
            rect2a_ = LeakyReLU(conv2a_)
            conv2b_ = ConvLayer(rect2a_, (gru_filters[1], 3, 3), params=conv2b.params) # 128 x 3 x 3
            rect2_ = LeakyReLU(conv2b_)
            conv2c_ = ConvLayer(pool1_, (gru_filters[1], 1, 1), params=conv2c.params) # 128 x 1 x 1
            res2_ = AddLayer(conv2c_, rect2_)
            pool2_ = PoolLayer(res2_)

            # Residual                               |=> -----------------=V
            # Conv -> leakyReLU -> Conv -> LeakyReLU -> Conv -> LeakyReLU -> MaxPooling
            conv3a_ = ConvLayer(pool2_, (gru_filters[2], 3, 3), params=conv3a.params) # 256 x 3 x 3
            rect3a_ = LeakyReLU(conv3a_)
            conv3b_ = ConvLayer(rect3a_, (gru_filters[2], 3, 3), params=conv3b.params) # 256 x 3 x 3
            rect3_ = LeakyReLU(conv3b_)
            conv3c_ = ConvLayer(pool2_, (gru_filters[2], 1, 1), params=conv3c.params) # 256 x 1 x 1
            res3_ = AddLayer(conv3c_, rect3_)
            pool3_ = PoolLayer(res3_)

            # Conv -> leakyReLU -> Conv -> LeakyReLU -> MaxPooling
            conv4a_ = ConvLayer(pool3_, (gru_filters[3], 3, 3), params=conv4a.params)  # 256 x 3 x 3
            rect4a_ = LeakyReLU(conv4a_)
            conv4b_ = ConvLayer(rect4a_, (gru_filters[3], 3, 3), params=conv4b.params)  # 256 x 3 x 3
            rect4_ = LeakyReLU(conv4b_)
            pool4_ = PoolLayer(rect4_)

            # Residual                               |=> -----------------=V
            # Conv -> leakyReLU -> Conv -> LeakyReLU -> Conv -> LeakyReLU -> MaxPooling
            conv5a_ = ConvLayer(pool4_, (gru_filters[4], 3, 3), params=conv5a.params)  # 256 x 3 x 3
            rect5a_ = LeakyReLU(conv5a_)
            conv5b_ = ConvLayer(rect5a_, (gru_filters[4], 3, 3), params=conv5b.params)  # 256 x 3 x 3
            rect5_ = LeakyReLU(conv5b_)
            conv5c_ = ConvLayer(pool4_, (gru_filters[4], 1, 1), params=conv5c.params)  # 256 x 1 x 1
            res5_ = AddLayer(conv5c_, rect5_)
            pool5_ = PoolLayer(res5_)

            # Residual                               |=> -----------------=V
            # Conv -> leakyReLU -> Conv -> LeakyReLU -> Conv -> LeakyReLU -> MaxPooling
            conv6a_ = ConvLayer(pool5_, (gru_filters[5], 3, 3), params=conv6a.params) # 256 x 3 x 3
            rect6a_ = LeakyReLU(conv6a_)
            conv6b_ = ConvLayer(rect6a_, (gru_filters[5], 3, 3), params=conv6b.params) # 256 x 3 x 3
            rect6_ = LeakyReLU(conv6b_)
            res6_ = AddLayer(pool5_, rect6_)
            pool6_ = PoolLayer(res6_)

            # Flatten Layer
            flat6_ = FlattenLayer(pool6_)

            # Fully connected layer
            fc7_ = TensorProductLayer(flat6_, fully_connecter_filter[0], params=fc7.params)
            rect7_ = LeakyReLU(fc7_)

            # h(t-1)
            prev_s_ = InputLayer(s_shape, prev_s_tensor)

            # FC layer convoluted with hidden states
            update_layer_ = FCConv3DLayer(
                prev_s_,
                rect7_, (gru_filters[1], gru_filters[1], 3, 3, 3), # 128 x 3 x 3 x 3
                params=update_layer.params)

            # FC layer convoluted with hidden states
            reset_layer_ = FCConv3DLayer(
                prev_s_,
                rect7_, (gru_filters[1], gru_filters[1], 3, 3, 3), # 128 x 3 x 3 x 3
                params=reset_layer.params)

            # Sigmoid( Wfx T(xt) (+) Uf * h(t-1) + bf )
            update_gate_ = SigmoidLayer(update_layer_)

            # 1 - u(t)
            compliment_update_gate_ = ComplementLayer(update_gate_)

            # Sigmoid (Wix T(xt) (+) Ui * h(t-1) + bi)
            reset_gate_ = SigmoidLayer(reset_layer_)

            # rt (.) h(t-1)
            rs_ = EltwiseMultiplyLayer(reset_gate_, prev_s_)

            # Uh * rt (.) h(t-1) + bh
            hidden_layer_ = FCConv3DLayer(
                rs_, rect7_, (gru_filters[1], gru_filters[1], 3, 3, 3), params=hidden_state_layer.params) # 128 x 3 x 3 x 3

            tanh_layer = TanhLayer(hidden_layer_)

            # ht = (1 - ut) (.) h(t-1) (+) tanh(  Uh * rt (.) h(t-1) + bh )
            gru_out_ = AddLayer(
                EltwiseMultiplyLayer(update_gate_, prev_s_),
                EltwiseMultiplyLayer(compliment_update_gate_, tanh_layer))

            return gru_out_.output, update_gate_.output


        s_update, _ = theano.scan(recurrence,
            sequences=[self.x],  # along with images, feed in the index of the current frame
            outputs_info=[tensor.zeros_like(np.zeros(s_shape),
                                            dtype=theano.config.floatX),
                           tensor.zeros_like(np.zeros(s_shape),
                                             dtype=theano.config.floatX)])

        # Update of all units
        update_all = s_update[-1]
        s_all = s_update[0]

        # Last hidden states. last timestep
        s_last = s_all[-1]


        ########## End GRU ##########


        ########## Decoder ##########

        # Depth of deconvolutional layers
        dcnn_filters = [128, 128, 128, 64, 32, 2]

        # Input Layer
        gru_s = InputLayer(s_shape, s_last)

        # Residual  |=>  ----------------------------------------------=V
        # Unpooling -> deconvolution -> LeakyReLU -> DeConv -> LeakyReLU ->
        unpool7 = Unpool3DLayer(gru_s)
        conv7a = Conv3DLayer(unpool7, (dcnn_filters[1], 3, 3, 3)) # 128 x 3 x 3 x 3
        rect7a = LeakyReLU(conv7a)
        conv7b = Conv3DLayer(rect7a, (dcnn_filters[1], 3, 3, 3)) # 128 x 3 x 3 x 3
        rect7 = LeakyReLU(conv7b)
        res7 = AddLayer(unpool7, rect7)

        # Residual  |=>  ----------------------------------------------=V
        # Unpooling -> deconvolution -> LeakyReLU -> DeConv -> LeakyReLU ->
        unpool8 = Unpool3DLayer(res7)
        conv8a = Conv3DLayer(unpool8, (dcnn_filters[2], 3, 3, 3)) # 128 x 3 x 3 x 3
        rect8a = LeakyReLU(conv8a)
        conv8b = Conv3DLayer(rect8a, (dcnn_filters[2], 3, 3, 3)) # 128 x 3 x 3 x 3
        rect8 = LeakyReLU(conv8b)
        res8 = AddLayer(unpool8, rect8)

        # Residual  |=>  ----------------------------------------------=V
        # Unpooling -> deconvolution -> LeakyReLU -> DeConv -> LeakyReLU ->
        unpool9 = Unpool3DLayer(res8)
        conv9a = Conv3DLayer(unpool9, (dcnn_filters[3], 3, 3, 3)) # 64 x 3 x 3 x 3
        rect9a = LeakyReLU(conv9a)
        conv9b = Conv3DLayer(rect9a, (dcnn_filters[3], 3, 3, 3)) # 64 x 3 x 3 x 3
        rect9 = LeakyReLU(conv9b)
        conv9c = Conv3DLayer(unpool9, (dcnn_filters[3], 1, 1, 1)) # 64 x 1 x 1 x 1
        res9 = AddLayer(conv9c, rect9)

        # Residual  |=>  ----------------------------------------------=V
        # Unpooling -> deconvolution -> LeakyReLU -> DeConv -> LeakyReLU ->
        conv10a = Conv3DLayer(res9, (dcnn_filters[4], 3, 3, 3)) # 32 x 3 x 3 x 3
        rect10a = LeakyReLU(conv10a)
        conv10b = Conv3DLayer(rect10a, (dcnn_filters[4], 3, 3, 3)) # 32 x 3 x 3 x 3
        rect10 = LeakyReLU(conv10b)
        conv10c = Conv3DLayer(rect10a, (dcnn_filters[4], 3, 3, 3)) # 32 x 3 x 3 x 3
        res10 = AddLayer(conv10c, rect10)

        # Last convolution
        conv11 = Conv3DLayer(res10, (dcnn_filters[5], 3, 3, 3)) # 2 x 3 x 3 x 3

        # Softmax layer
        softmax_loss = SoftmaxWithLoss3D(conv11.output)


        ########## End Decoder #########

        self.loss = softmax_loss.loss(self.y)
        self.error = softmax_loss.error(self.y)
        self.params = get_trainable_params()
        self.output = softmax_loss.prediction()
        self.activations = [update_all]


    # Save the weights to a file
    def save(self, filename):
        params_cpu = []
        for param in self.params:
            params_cpu.append(param.val.get_value())
        np.save(filename, params_cpu)
        print('saving network parameters to ' + filename)

    # Load parameters from file
    def load(self, filename, ignore_param=True):
        print('loading network parameters from ' + filename)
        params_cpu_file = np.load(filename)
        if filename.endswith('npz'):
            params_cpu = params_cpu_file[params_cpu_file.keys()[0]]
        else:
            params_cpu = params_cpu_file

        succ_ind = 0
        for param_idx, param in enumerate(self.params):
            try:
                param.val.set_value(params_cpu[succ_ind])
                succ_ind += 1
            except IndexError:
                if ignore_param:
                    print('Ignore mismatch')
                else:
                    raise
