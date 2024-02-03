import torch
from models.wavenet import WaveNet

class VoiceMimic:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._init_model()

    def _init_model(self):
        model = WaveNet(num_classes=self.config.num_classes,
                        num_layers=self.config.num_layers,
                        num_blocks=self.config.num_blocks,
                        kernel_size=self.config.kernel_size,
                        residual_channels=self.config.residual_channels,
                        dilation_channels=self.config.dilation_channels,
                        skip_channels=self.config.skip_channels,
                        output_channels=self.config.output_channels)
        model.to(self.device)
        return model

    def load_weights(self, weights_path):
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def synthesize_voice(self, input_sequence):
        with torch.no_grad():
            input_sequence = input_sequence.to(self.device)
            output = self.model.generate(input_sequence)
        return output
