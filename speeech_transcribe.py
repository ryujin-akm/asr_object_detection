

device = torch.device('cpu')   # you can use any pytorch device
models = OmegaConf.load('models.yml')




# wav to text method
def wav_to_text(f='test.wav'):
  batch = read_batch([f])
  input = prepare_model_input(batch, device=device)
  output = model(input)
  return decoder(output[0].cpu())

# Transcribe

#@markdown { run: "auto" }

language = "English" #@param ["English", "German", "Spanish"]

print(language)
if language == 'German':
  model, decoder = init_jit_model(models.stt_models.de.latest.jit, device=device)
elif language == "Spanish":
  model, decoder = init_jit_model(models.stt_models.es.latest.jit, device=device)
else:
  model, decoder = init_jit_model(models.stt_models.en.latest.jit, device=device)


#@markdown { run: "auto" }

use_VAD = "No" #@param ["Yes", "No"]

#@markdown Either record audio from microphone or upload audio from file (.mp3 or .wav) { run: "auto" }

record_or_upload = "Record" #@param ["Record", "Upload (.mp3 or .wav)"]
record_seconds =   4#@param {type:"number", min:1, max:10, step:1}
sample_rate = 16000

def _apply_vad(audio, boot_time=0, trigger_level=9, **kwargs):
  print('\nVAD applied\n')
  vad_kwargs = dict(locals().copy(), **kwargs)
  vad_kwargs['sample_rate'] = sample_rate
  del vad_kwargs['kwargs'], vad_kwargs['audio']
  audio = vad(torch.flip(audio, ([0])), **vad_kwargs)
  return vad(torch.flip(audio, ([0])), **vad_kwargs)

def _recognize(audio):
  display(Audio(audio, rate=sample_rate, autoplay=True))
  if use_VAD == "Yes":
    audio = _apply_vad(audio)
  wavfile.write('test.wav', sample_rate, (32767*audio).numpy().astype(np.int16))
  transcription = wav_to_text()
  print('\n\nTRANSCRIPTION:\n')
  print(transcription)

def _record_audio(b):
  clear_output()
  audio = record_audio(record_seconds)
  wavfile.write('recorded.wav', sample_rate, (32767*audio).numpy().astype(np.int16))
  _recognize(audio)

def _upload_audio(b):
  clear_output()
  audio = upload_audio()
  _recognize(audio)
  return audio

if record_or_upload == "Record":
  button = widgets.Button(description="Record Speech")
  button.on_click(_record_audio)
  display(button)
else:
  audio = _upload_audio("")


#@markdown Check audio after applying VAD { run: "auto" }

if record_or_upload == "Record":
  audio = read_audio('recorded.wav', sample_rate)
display(Audio(_apply_vad(audio), rate=sample_rate, autoplay=True))
