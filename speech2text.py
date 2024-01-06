from transformers import pipeline
import datasets
import soundfile as sf
from IPython.display import Audio

# Load the automatic-speech-recognition pipeline
asr = pipeline("automatic-speech-recognition")

# Load the dataset
dataset = datasets.load_dataset("lm.ckpt")

# Define a function to map the dataset
def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

# Apply the mapping function to the dataset
dataset = dataset.map(map_to_array)

# Display the audio using IPython
display(Audio(dataset[0]['speech'], rate=16000))

# Set the dataset format to numpy
dataset.set_format("numpy")

# Perform ASR on the audio
pred = asr(dataset[0]["speech"], model_name="your_model_name")
print(pred)
