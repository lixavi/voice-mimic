import torch
from models.wavenet import WaveNet

class VoiceMimic:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._init_model()



    def load_weights(self, weights_path):
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def synthesize_voice(self, input_sequence):
        with torch.no_grad():
            input_sequence = input_sequence.to(self.device)
            output = self.model.generate(input_sequence)
        return output
