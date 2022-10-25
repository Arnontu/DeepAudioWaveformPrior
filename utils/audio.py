from pathlib import Path
import numpy as np
import torch
import torchaudio

from utils.metrics import snr

EPS = np.finfo(float).eps


###################################################################################################################
###################################################################################################################

def mono_resample_np(wav, sr, logger, target_sr=16000):
    if len(wav.size()) > 1:
        wav = wav.mean(0)
    if sr != target_sr:
        logger.warning(f"Resampling audio from {sr} to {target_sr}")
        wav = (torchaudio.transforms.Resample(sr, target_sr)(wav))
    return wav.numpy()


def split_given_size(a, size):
    res = np.split(a, np.arange(size, len(a), size,dtype=np.int64))
    if len(a) / size != len(a) // size:
        # remove reminder
        res = res[:-1]
    return np.array(res)


def filter_quiet(tracks, thresh=.1):
    indices = np.arange(len(tracks))
    to_keep = np.where(((tracks == 0).sum(1) / tracks.shape[-1]) <= thresh)[0]
    return tracks[to_keep], indices[to_keep]


def sample_noise(args, shape):
    if args.noise_class == "GAUSSIAN":
        return np.random.normal(0, args.noise_std, shape)
    if args.noise_class == "UNIFORM":
        return np.random.uniform(-1, 1, shape)
    assert 'Invalid noise type'


def process_audio(args, audio, audio_sr, logger, chunk_index=None):
    audio = mono_resample_np(audio, audio_sr, logger, args.samplerate)
    audio = audio[int(args.trim_start * args.samplerate):]  # cut off 0.5 sec first
    if args.clip_length:
        audio_chunks = split_given_size(audio, args.clip_length * args.samplerate)
    else:
        audio_chunks = np.array([audio])
    if not audio_chunks.any():
        return None, None
    if chunk_index is None:
        audio_chunks, audio_chunks_indices = filter_quiet(audio_chunks, args.quiet_thresh)
        if not audio_chunks.any():
            return None, None
        chunk_index = np.random.choice(len(audio_chunks))
    audio = audio_chunks[chunk_index]
    return audio, chunk_index


###################################################################################################################
###################################################################################################################
# Section code borrowed from https://github.com/microsoft/DNS-Challenge

def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)


def normalize(audio, target_level=-25):
    '''Normalize the signal to the target level'''
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def snr_mixer(clean, noise, snr_val, target_level_lower=-35, target_level_upper=-15, target_level=-25,
              clipping_threshold=0.99):
    '''Function to mix clean speech and noise at various SNR levels'''

    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean) - len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise) - len(clean)))

    # Normalizing to -25 dB FS
    clean = clean / (max(abs(clean)) + EPS)
    clean = normalize(clean, target_level)
    rmsclean = (clean ** 2).mean() ** 0.5

    noise = noise / (max(abs(noise)) + EPS)
    noise = normalize(noise, target_level)
    rmsnoise = (noise ** 2).mean() ** 0.5

    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10 ** (snr_val / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel

    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = np.random.randint(target_level_lower, target_level_upper)
    rmsnoisy = (noisyspeech ** 2).mean() ** 0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy

    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel / noisyspeech_maxamplevel
        noisy_rms_level = int(20 * np.log10(scalarnoisy / noisyspeech_maxamplevel * (rmsnoisy + EPS)))

    return clean, noisenewlevel, noisyspeech, noisy_rms_level


###################################################################################################################
###################################################################################################################

def get_clean_n_noisy(paths, args, logger):
    clean_path, noisy_path = paths.strip().split(",")
    try:
        clean, sr_clean = torchaudio.load(clean_path)
        logger.info(f"Loading clean audio from {clean_path}")
    except:
        logger.error(f"Failed loading audio: {clean_path}")
        return None
    clean_audio, chunk_index = process_audio(args, clean, sr_clean, logger)
    if clean_audio is None:
        logger.error(f"Preprocess skipped: {clean_path}")
        return None

    if noisy_path:
        try:
            noisy, sr_noisy = torchaudio.load(noisy_path)
            logger.info(f"Loading noisy audio from {noisy_path}")
        except:
            logger.error(f"Failed loading audio: {noisy_path}")
            return None
        assert sr_noisy == sr_clean == args.samplerate, f"Samplesrates doesnt match! Required samplerate: {args.samplerate}. Clean audio samplerate: {sr_clean}. Noisy audio samplerate: {sr_noisy}."
        assert clean.shape[-1] == noisy.shape[
            -1], f"Mismatch length! clean audio :{clean.shape[-1]}, noisy audio: {noisy.shape}"
        noisy_audio, _ = process_audio(args, noisy, sr_noisy, logger, chunk_index)
        loaded_snr = snr(torch.from_numpy(clean_audio), torch.from_numpy(noisy_audio))
        f = logger.info if (loaded_snr - args.snr).abs() < 1e-3 else logger.warning
        f(f"Loaded noisy audio with SNR {loaded_snr:.2f}. Specified SNR in args is {args.snr:.2f}")
        # logger.info
        # else:
        #     logger.warning(f"Loaded noisy audio with snr {loaded_snr:.2f}, SNR specified in args is {args.snr:.2f}")

    else:
        logger.info("No noisy path, sampling noise by args")
        noise = sample_noise(args, clean_audio.shape)
        clean_audio, _, noisy_audio, _ = snr_mixer(clean_audio, noise, args.snr)

    return torch.from_numpy(clean_audio).unsqueeze(0).unsqueeze(0).to(args.device), \
           torch.from_numpy(noisy_audio).unsqueeze(0).unsqueeze(0).to(args.device), clean_path

###################################################################################################################
###################################################################################################################
