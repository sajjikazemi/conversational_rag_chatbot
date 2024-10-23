from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import sounddevice as sd
import numpy as np
import torch

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device='cuda')
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
speech = synthesiser("Hi to sajji an artificial intelligence engineer master", forward_params={"speaker_embeddings": speaker_embedding})
sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

audio_data = np.array(speech["audio"])
sd.play(audio_data, speech["sampling_rate"])
sd.wait()
