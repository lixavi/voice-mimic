class Config:
    def __init__(self):
        # Model configuration
        self.num_classes = 256  # Number of possible audio values (e.g., 8-bit audio has 256 possible values)
        self.num_layers = 10  # Number of layers in each block
        self.num_blocks = 3  # Number of blocks in the WaveNet architecture
        self.kernel_size = 2  # Size of the convolutional kernel
        self.residual_channels = 64  # Number of channels in the residual connections
        self.dilation_channels = 128  # Number of channels in the dilated convolutions
        self.skip_channels = 256  # Number of channels in the skip connections
        self.output_channels = 256  # Number of output channels in the final convolutional layer

        # Training configuration
        self.sequence_length = 100  # Length of input sequence
        self.batch_size = 32  # Batch size for training
        self.num_epochs = 10  # Number of training epochs
        self.learning_rate = 0.001  # Learning rate for training

        # Dataset configuration
        self.data_dir = 'data/'  # Directory containing the training data
        self.train_file = 'train_data.npy'  # File name of the training data

config = Config()
