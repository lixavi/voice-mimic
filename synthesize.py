import torch
from models.wavenet import WaveNet

def synthesize(model, input_sequence, length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_sequence = input_sequence.unsqueeze(0).to(device)
    generated_sequence = model.generate(input_sequence, length=length)
    return generated_sequence.squeeze(0).cpu().numpy()

def main():
    # Set the path to your trained WaveNet model
    model_path = 'path/to/your/trained/model.pth'

    # Load the trained model
    model = WaveNet(your_model_parameters)
    model.load_state_dict(torch.load(model_path))
    

