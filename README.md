# VoiceMimic
Voice cloning system employing WaveNet, a deep generative model of raw audio waveforms for voice synthesis.

## Overview

VoiceMimic utilizes the WaveNet architecture to synthesize realistic human-like voices. It takes an input audio sequence and generates a new sequence of audio samples that mimic the input voice. The system can be trained on a dataset of audio recordings to learn the characteristics of a particular voice and then generate new audio samples that resemble the input voice.

## Components

The VoiceMimic system consists of the following components:

1. `main.py`: The main entry point of the program. It initializes the VoiceMimic system, trains the WaveNet model (optional), loads pre-trained weights, and synthesizes voice.

2. `voice_mimic.py`: Contains the implementation of the VoiceMimic system, including model initialization, loading weights, and synthesizing voice.

3. `wavenet.py`: Implementation of the WaveNet model, including residual blocks and generation logic.

4. `dataset.py`: Dataset preparation and processing for training the WaveNet model.

5. `train.py`: Training script for the WaveNet model.

6. `synthesize.py`: Script for synthesizing voice using the trained WaveNet model.

7. `utils.py`: Utility functions for handling audio data, such as saving and loading audio files, and normalizing audio data.

8. `config.py`: Configuration parameters for the VoiceMimic system, including model architecture settings, training parameters, and dataset configuration.

9. `requirements.txt`: List of required Python packages for running the VoiceMimic system.
