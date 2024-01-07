from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch

# Load the tokenizer and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Define a function to transcribe speech
def transcribe_audio(audio_file):
    try:
        speech_input, _ = torchaudio.load(audio_file)
        input_values = processor(speech_input[0].numpy(), return_tensors="pt").input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription
    except Exception as e:
        return str(e)

# Audio file path (assuming it's in the same directory as the script)
audio_file = "namibia.wav"  # Replace with the correct file path
result = transcribe_audio(audio_file)

# Save the transcription to a .txt file
output_file = "transcription.txt"
with open(output_file, "w") as text_file:
    text_file.write(result)

print("Transcription saved to", output_file)
