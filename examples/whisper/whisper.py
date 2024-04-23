import os
import sys
import time
import numpy as np
from dataclasses import dataclass
from .tokenizer import get_tokenizer

import ctypes
# Load this here explicitly into the process, so importing pybind_whisper
# doesn't have to search for librknnrt.so.
_ = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'librknnrt.so'))

from .pybind_whisper import WhisperModel as CWhisperModel

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def set_encoder_params(model, p, dims):
    Wconv0, conv0_bias = p['encoder.conv1.weight'], p['encoder.conv1.bias']
    Wconv1, conv1_bias = p['encoder.conv2.weight'], p['encoder.conv2.bias']
    # [output_depth, channels, input_depth]
    model.set_conv0_weights(Wconv0.transpose([0, 2, 1]))
    model.set_conv1_weights(Wconv1.transpose([0, 2, 1]))
    model.set_conv0_bias(conv0_bias)
    model.set_conv1_bias(conv1_bias)

    model.set_encoder_positional_embedding(p['encoder.positional_embedding'])

    for i in range(dims.n_audio_layer):
        prefix = f'encoder.blocks.{i}'
        model.set_encoder_attn_ln_gamma(i, p[f'{prefix}.attn_ln.weight'])
        model.set_encoder_attn_ln_beta(i, p[f'{prefix}.attn_ln.bias'])
        model.set_encoder_attn_Wq(i, p[f'{prefix}.attn.query.weight'].T)
        model.set_encoder_attn_q_bias(i, p[f'{prefix}.attn.query.bias'])
        model.set_encoder_attn_Wk(i, p[f'{prefix}.attn.key.weight'].T)
        model.set_encoder_attn_Wv(i, p[f'{prefix}.attn.value.weight'].T)
        model.set_encoder_attn_v_bias(i, p[f'{prefix}.attn.value.bias'])
        model.set_encoder_attn_Wout(i, p[f'{prefix}.attn.out.weight'].T)
        model.set_encoder_attn_out_bias(i, p[f'{prefix}.attn.out.bias'])

        model.set_encoder_mlp_ln_gamma(i, p[f'{prefix}.mlp_ln.weight'])
        model.set_encoder_mlp_ln_beta(i, p[f'{prefix}.mlp_ln.bias'])
        model.set_encoder_Wfc1(i, p[f'{prefix}.mlp.0.weight'].T)
        model.set_encoder_fc1_bias(i, p[f'{prefix}.mlp.0.bias'])
        model.set_encoder_Wfc2(i, p[f'{prefix}.mlp.2.weight'].T)
        model.set_encoder_fc2_bias(i, p[f'{prefix}.mlp.2.bias'])

    model.set_encoder_ln_post_gamma(p[f'encoder.ln_post.weight'])
    model.set_encoder_ln_post_beta(p[f'encoder.ln_post.bias'])


def next_multiple_of_3(x):
    return int(3 * ((x + 2) // 3))


def set_decoder_params(model, p, dims):
    model.set_decoder_positional_embedding(p['decoder.positional_embedding'])

    for i in range(dims.n_text_layer):
        prefix = f'decoder.blocks.{i}'
        model.set_decoder_attn_ln_gamma(i, p[f'{prefix}.attn_ln.weight'])
        model.set_decoder_attn_ln_beta(i, p[f'{prefix}.attn_ln.bias'])
        model.set_decoder_attn_Wq(i, p[f'{prefix}.attn.query.weight'].T)
        model.set_decoder_attn_q_bias(i, p[f'{prefix}.attn.query.bias'])
        model.set_decoder_attn_Wk(i, p[f'{prefix}.attn.key.weight'].T)
        model.set_decoder_attn_Wv(i, p[f'{prefix}.attn.value.weight'].T)
        model.set_decoder_attn_v_bias(i, p[f'{prefix}.attn.value.bias'])
        model.set_decoder_attn_Wout(i, p[f'{prefix}.attn.out.weight'].T)
        model.set_decoder_attn_out_bias(i, p[f'{prefix}.attn.out.bias'])

        model.set_decoder_cross_attn_ln_gamma(i, p[f'{prefix}.cross_attn_ln.weight'])
        model.set_decoder_cross_attn_ln_beta(i, p[f'{prefix}.cross_attn_ln.bias'])
        model.set_decoder_cross_attn_Wq(i, p[f'{prefix}.cross_attn.query.weight'].T)
        model.set_decoder_cross_attn_q_bias(i, p[f'{prefix}.cross_attn.query.bias'])
        model.set_decoder_cross_attn_Wk(i, p[f'{prefix}.cross_attn.key.weight'].T)
        model.set_decoder_cross_attn_Wv(i, p[f'{prefix}.cross_attn.value.weight'].T)
        model.set_decoder_cross_attn_v_bias(i, p[f'{prefix}.cross_attn.value.bias'])
        model.set_decoder_cross_attn_Wout(i, p[f'{prefix}.cross_attn.out.weight'].T)
        model.set_decoder_cross_attn_out_bias(i, p[f'{prefix}.cross_attn.out.bias'])

        model.set_decoder_mlp_ln_gamma(i, p[f'{prefix}.mlp_ln.weight'])
        model.set_decoder_mlp_ln_beta(i, p[f'{prefix}.mlp_ln.bias'])
        model.set_decoder_Wfc1(i, p[f'{prefix}.mlp.0.weight'].T)
        model.set_decoder_fc1_bias(i, p[f'{prefix}.mlp.0.bias'])
        model.set_decoder_Wfc2(i, p[f'{prefix}.mlp.2.weight'].T)
        model.set_decoder_fc2_bias(i, p[f'{prefix}.mlp.2.bias'])

    model.set_decoder_ln_gamma(p[f'decoder.ln.weight'])
    model.set_decoder_ln_beta(p[f'decoder.ln.bias'])
    Wdetokenizer = p[f'decoder.token_embedding.weight'].T
    n_vocab = next_multiple_of_3(dims.n_vocab)
    slice_len = n_vocab // 3
    model.set_detokenizer0(Wdetokenizer[:, :slice_len])
    model.set_detokenizer1(Wdetokenizer[:, slice_len:2*slice_len])
    extra_columns = n_vocab - dims.n_vocab
    third_slice = Wdetokenizer[:, 2*slice_len:]
    if extra_columns:
        third_slice = np.concatenate([third_slice, np.zeros_like(third_slice[:, :extra_columns])], -1)
    model.set_detokenizer2(third_slice)


class WhisperModel(object):
    def __init__(self, model='tiny.en', verbose=True):
        params_file = os.path.join(os.path.dirname(__file__), "weights", f"{model}.npz")
        params_file = np.load(params_file)
        dims = {k.split('/')[-1]:v for k, v in params_file.items() if k.startswith('dims/')}
        params = {k.split('/')[-1]:v for k, v in params_file.items() if k.startswith('params/')}

        dims = ModelDimensions(**dims)
        n_vocab = next_multiple_of_3(dims.n_vocab)
        self.model = CWhisperModel(dims.n_mels,
                                   dims.n_audio_ctx,
                                   dims.n_audio_state,
                                   dims.n_audio_head,
                                   dims.n_audio_layer,
                                   dims.n_text_ctx,
                                   dims.n_text_state,
                                   dims.n_text_head,
                                   dims.n_text_layer,
                                   n_vocab)
        self.dims = dims

        set_encoder_params(self.model, params, dims)
        set_decoder_params(self.model, params, dims)
        self.multilingual = not model.endswith('.en')
        self.tokenizer = get_tokenizer(multilingual=self.multilingual)
        self.lang_dict = dict(zip(self.tokenizer.all_language_codes, self.tokenizer.all_language_tokens))
        assert os.sched_getaffinity(os.getpid()) == set([4, 5, 6, 7]), (
            f'Should be run with taskset -c4-7')

        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        self.N_FFT = 400
        self.HOP_LENGTH = 160
        self.mel_filters = np.load(os.path.join(assets_dir, 'mel_filters.npz'))['mel_80'].T
        fft_matrix_file = np.load(os.path.join(assets_dir, 'fft_params.npz'))
        self.fft_matrix_real = fft_matrix_file['fft_matrix_real']
        self.fft_matrix_imag = fft_matrix_file['fft_matrix_imag']

        self.verbose = verbose

    def mel_spectrogram(self, audio):
        audio = audio.squeeze()
        audio = np.pad(audio, [self.N_FFT // 2, self.N_FFT // 2], mode='reflect')
        audio = np.lib.stride_tricks.sliding_window_view(audio, self.N_FFT)[::self.HOP_LENGTH]
        stft_real = audio @ self.fft_matrix_real.squeeze()
        stft_imag = audio @ self.fft_matrix_imag.squeeze()
        magnitudes = stft_real ** 2 + stft_imag ** 2
        mel_spec = magnitudes @ self.mel_filters
        log_spec = np.clip(mel_spec, 1e-10, np.finfo(np.float32).max)
        log_spec = np.log10(log_spec)
        log_spec = np.maximum(log_spec, np.max(log_spec) - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        log_spec = log_spec[:-1, :][np.newaxis, ...]
        return log_spec

    def decode_no_timestamps(self, mel, task='transcribe', src_lang='en'):
        tokenizer = self.tokenizer
        suppress_tokens_sans_no_speech = [
            tokenizer.transcribe,
            tokenizer.translate,
            tokenizer.sot,
            tokenizer.sot_prev,
            tokenizer.sot_lm,
            tokenizer.no_speech,
            tokenizer.no_timestamps] + list(tokenizer.non_speech_tokens)
        # Suppress padding tokens.
        for i in range(self.dims.n_vocab, next_multiple_of_3(self.dims.n_vocab)):
          suppress_tokens_sans_no_speech += [i]

        suppress_tokens = suppress_tokens_sans_no_speech + [tokenizer.no_speech]
        initial_suppress_tokens = suppress_tokens + tokenizer.encode(' ') + [tokenizer.eot]

        self.model.reset(mel)

        initial_prompt = list(tokenizer.sot_sequence_including_notimestamps)
        if self.multilingual:
            assert src_lang in self.lang_dict, f'{src_lang} is not a supported language'
            initial_prompt[1] = self.lang_dict[src_lang]
            if task == 'translate': initial_prompt[2] = self.tokenizer.translate
        if self.verbose:
            print(f'{initial_prompt} {self.tokenizer.decode(initial_prompt)}')
        for p in initial_prompt:
            self.model.call_no_copy(p)

        logprobs = self.model.log_softmax(initial_suppress_tokens).view(np.float16)

        decoded_tokens = [np.argmax(logprobs)]

        while len(decoded_tokens) < 224:
            self.model.call_no_copy(decoded_tokens[-1])
            logprobs = self.model.log_softmax(suppress_tokens).view(np.float16)
            speech_token = np.argmax(logprobs[:tokenizer.eot])
            speech_logprob = logprobs[speech_token]
            eot_logprob = logprobs[tokenizer.eot]

            if eot_logprob > speech_logprob:
                break
            decoded_tokens.append(speech_token)
        return decoded_tokens


def decode_wav_file(filename, model='tiny.en', task='transcribe', src_lang='en'):
    import wave
    w = wave.open(filename)
    assert w.getnchannels() == 1, f'Only one channel supported'
    assert w.getsampwidth() == 2, f'Datatype should be int16'
    assert w.getframerate() == 16000, f'Only 16kHz supported'
    frames = w.readframes(w.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16)
    model = WhisperModel(model)
    return decode_pcm(audio, model, task, src_lang)

def decode_pcm(audio, model, task='transcribe', src_lang='en'):
    import tqdm
    assert type(audio) == np.ndarray, f'audio should be a numpy array'
    if audio.dtype in (np.int16, np.int32, np.int8):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    elif audio.dtype == np.float64:
        audio = audio.astype(np.float32)
    if type(model) is str:
        model = WhisperModel(model)
    assert type(model) is WhisperModel, f'model should be a WhisperModel or a string'
    segments = np.split(audio, np.arange(0, audio.shape[0], 480000)[1:])
    decoded = []
    for segment in tqdm.tqdm(segments):
        remainder = 480000 - segment.shape[0]
        segment = np.concatenate([segment, np.zeros([remainder]).astype(np.float32)])
        mel = model.mel_spectrogram(segment[np.newaxis])
        tokens = model.decode_no_timestamps(mel, task, src_lang)
        decoded += tokens
    return model.tokenizer.decode(decoded)
