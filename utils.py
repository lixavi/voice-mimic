
    """
    Save audio data to a file.

    Args:
    - filename (str): The name of the file to save.
    - audio_data (ndarray): The audio data to save.
    - sample_rate (int): The sample rate of the audio data.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    sf.write(filename, audio_data, sample_rate)

def load_audio(filename):
    """
    Load audio data from a file.

    Args:
    - filename (str): The name of the file to load.

    Returns:
    - audio_data (ndarray): The loaded audio data.
    - sample_rate (int): The sample rate of the loaded audio data.
    """
    audio_data, sample_rate = sf.read(filename, dtype='float32')
    return audio_data, sample_rate

def normalize_audio(audio_data):
    """
    Normalize audio data to the range [-1, 1].

    Args:
    - audio_data (ndarray): The audio data to normalize.

    Returns:
    - normalized_audio (ndarray): The normalized audio data.
    """
    max_value = np.max(np.abs(audio_data))
    normalized_audio = audio_data / max_value if max_value > 0 else audio_data
    return normalized_audio
