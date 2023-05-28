# from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
import librosa
from torch import nn
import torchaudio.transforms as T

device = "cuda:1" if torch.cuda.is_available() else "cpu"
# loading our model weights
model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(device)
# loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)

wav_pth = "/cfs3/share/corpus/eval_dataset/test_codec/source/司徒兰芳 - 爱的草原情的河.mp3.wav"
wav, sampling_rate = librosa.load(wav_pth, sr=24000)
chunk_size = sampling_rate * 60  # 10 seconds

hidden_state_list = []

for n in range(0, len(wav) // chunk_size + 1):
    if (n + 1) * chunk_size > len(wav):
        cut_wav = wav[n * chunk_size:]
    else:
        cut_wav = wav[n * chunk_size:(n + 1) * chunk_size]

    resample_rate = processor.sampling_rate
    # make sure the sample_rate aligned
    if resample_rate != sampling_rate:
        print(f'setting rate from {sampling_rate} to {resample_rate}')
        resampler = T.Resample(sampling_rate, resample_rate)
    else:
        resampler = None

    # audio file is decoded on the fly
    if resampler is None:
        input_audio = cut_wav
    else:
        input_audio = resampler(torch.from_numpy(cut_wav))

    inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")
    print("input shape: ", inputs.input_values.shape)

    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)


    # take a look at the output shape, there are 25 layers of representation
    # each layer performs differently in different downstream tasks, you should choose empirically
    all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
    print(all_layer_hidden_states.shape)  # [25 layer, Time steps, 1024 feature_dim]
    print(all_layer_hidden_states[20].shape)  # [25 layer, Time steps, 1024 feature_dim]
    hidden_state_list.append(all_layer_hidden_states[20])

    # for utterance level classification tasks, you can simply reduce the representation in time
    time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
    print(time_reduced_hidden_states.shape)  # [25, 1024]

    # you can even use a learnable weighted average representation
    aggregator = nn.Conv1d(in_channels=25, out_channels=1, kernel_size=1).to(device)
    weighted_avg_hidden_states = aggregator(time_reduced_hidden_states.unsqueeze(0)).squeeze()
    print(weighted_avg_hidden_states.shape)  # [1024]
    print(time_reduced_hidden_states[20].shape)

total_layer20_hidden_states = torch.cat(hidden_state_list, dim=0)
print(total_layer20_hidden_states.shape)
