import argparse
import logging
import shutil
import torch
import numpy as np
from pathlib import Path
from scipy.io import wavfile
import json
from utils.plot import _plot_wavs, _plot_wav_to_stft, _plot_loss, _plot_metric


###################################################################################################################
###################################################################################################################



def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_args():
    parser = argparse.ArgumentParser("DAWP", description="Train and evaluate Deep Audio Waveform Prior.")
    parser.add_argument("--experiment_repetition", type=int, default=1,
                        help="Number of repetition of the given setup(=experiment)")
    parser.add_argument("-e", "--epochs", type=int, default=5000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--show_every", type=int, default=50, help="Epochs between plots")

    parser.add_argument("--save_outputs", action="store_true", default=False, dest="save_outputs",
                        help="Save wavs, waves and spectrograms of outputs")
    parser.add_argument("--show_estimates_vs_noise", "--show_overfit", action="store_true", default=False,
                        dest="show_estimates_vs_noise", help="In metrics plot add graph of metric(estimate,noisy)")
    parser.add_argument("--source", required=True, type=Path, help="Path to inputs")
    parser.add_argument("--output_dir", type=Path, default="./evals", help="Path for experiments output")
    parser.add_argument("--device", type=str, default="cuda", choices=['cuda', 'cpu'])

    parser.add_argument("--noise_class", type=str, default="GAUSSIAN", choices=['GAUSSIAN', 'UNIFORM'])
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--crit", type=str, default="L1", choices=['L1', 'L2'])
    parser.add_argument("--snr", type=float, default=2.5, help="SNR")
    parser.add_argument("--samplerate", type=int, default='16000')


    parser.add_argument("--skip", action="store_true", default=False, dest="skip", help="Add skip connections")
    parser.add_argument("--attention_layers", type=int, default=0, help="Number of attention layers")
    parser.add_argument("--attention_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--depth", type=int, default=6, help="Number of layers for the encoder and decoder")
    parser.add_argument("--lstm_layers", type=int, default=2, help="Number of layers for the LSTM")
    parser.add_argument("--glu", action="store_true", default=False, dest="glu",
                        help="Replace all GLUs by ReLUs")
    parser.add_argument("--resample", action="store_false", default=True, dest="resample",
                        help="Resample input BxCxT -> BxCx2T and vice verse for output")


    parser.add_argument("--clip_length", type=float, default=2,
                        help="Clip inputs to <clip_length> seconds. 0 for no clipping.")
    parser.add_argument("--trim_start", type=float, default=0,
                        help="Trim <trim_start> second from the beginning of the input audio.")
    parser.add_argument("--quiet_thresh", type=float, default=.2,
                        help="Skip inputs if <quiet_thresh> of the input is silent")

    return parser.parse_args()


def get_crit(args):
    if args.crit == "L1":
        return torch.nn.L1Loss()
    if args.crit == "L2":
        return torch.nn.MSELoss()


def get_paths(source):
    """
    Extract input paths from source.
    Source path can be 2 options:
    1) Directory: extract (recursively) all <files> paths which considered clean audio paths.
    2) File: Expect each line to start with clean_path and optionaly to include noisy path, where paths are
             separated with ","
    e.g
    <clean_path_1>,<noisy_path_1>
    <clean_path_2>
    <clean_path_3>
    <clean_path_4>,<noisy_path_4>

    :param source: Path to directory/file
    :return: Input paths extracted from source
    """
    if source.is_dir():
        paths = [f"{path}, " for path in source.rglob("*") if path.is_file()]
    else:
        with open(source, 'r') as f:
            paths = f.readlines()
    np.random.shuffle(paths)
    return paths


###################################################################################################################
###################################################################################################################

def update_state(clean, noisy, basedir, epoch, losses, metrics_tracker, out, args):
    for metric in metrics_tracker:
        # clean_target,noisy_target
        metric.clean(out)
        metric.noisy(out)
    if epoch % args.show_every == 0:
        if args.save_outputs:
            wavs = torch.stack(tuple([noisy, out, clean])).detach().cpu().numpy()
            _plot_wavs(wavs, sr=args.samplerate, prefix=str(basedir / "img_wavs" / f"{epoch:05d}"))
            _plot_wav_to_stft(wavs, sr=args.samplerate, prefix=str(basedir / "img_mel" / f"{epoch:05d}"))
            wavfile.write(str(basedir / "wavs" / f"epoch_{epoch:05d}_output.wav"), rate=args.samplerate,
                          data=out.squeeze().detach().cpu().numpy())
        _plot_loss(basedir, losses)
        for metric in metrics_tracker:
            _plot_metric(metric, basedir, show_estimates_vs_noise=args.show_estimates_vs_noise)


def save_repetition_results(metric_tracker, loss, samplerate, basedir):
    best = {}
    for metric in metric_tracker:
        wavfile.write(str(basedir / f"{metric}_best.wav"), rate=samplerate,
                      data=metric.get_wav())
        best[str(metric)] = {"value": f"{metric.best_clean:.4f}",
                             "epoch": f"{metric.best_clean_epoch}",
                             "reference": f"{metric.get_ref():.4f}"
                             }
        np.save(basedir / f"{metric}_estimates_vs_clean.npy", metric.get_clean())
        np.save(basedir / f"{metric}_estimates_vs_noisy.npy", metric.get_noisy())
    np.save(basedir / f"loss.npy", loss)
    with open(str((basedir) / "metrics_best.json"), "w") as f:
        f.write((json.dumps(best, indent=4)))


def save_experiment_results(experiment_path, logger, repetitions_to_ignore=()):
    results = {}
    for path in experiment_path.iterdir():
        if path.is_dir() and path.name not in repetitions_to_ignore:
            try:
                with open(str((path) / "metrics_best.json"), "r") as f:
                    metrics_best = json.load(f)
                    for metric in metrics_best:
                        if metric in results:
                            results[str(metric)][path.name] = metrics_best[metric]
                        else:
                            results[str(metric)] = {path.name: metrics_best[metric]}
            except:
                logger.error(f"Check 'metrics_best.json' file within: {path}")
    with open(str(experiment_path / "metrics_best.json"), 'w') as f:
        f.write((json.dumps(results, indent=4)))
    with open(str(experiment_path / "metrics_best_stat.json"), 'w') as f:
        vals = {metric: np.array([results[metric][rep]['value'] for rep in results[metric]]).astype(np.float) for
                metric in results}
        ref = {metric: np.array([results[metric][rep]['reference'] for rep in results[metric]]).astype(np.float) for
               metric in results}
        stat = {metric: {'mean': f"{vals[metric].mean():.4f}",
                         'std': f"{vals[metric].std():.4f}",
                         'delta mean': f"{(vals[metric] - ref[metric]).mean():.4f}",
                         'repetitions': len(vals[metric])} for metric in vals}
        f.write((json.dumps(stat, indent=4)))


###################################################################################################################
###################################################################################################################

def get_experiment_name(args):
    """
    Return the name of an experiment given the args. Some parameters are ignored,
    for instance --workers, as they do not impact the final result.
    """
    include_args = ['depth', 'skip', 'glu', 'lstm_layers', 'attention_layers', 'snr', 'noise_class', 'noise_std',
                    'crit']
    parts = []
    name_args = dict(args.__dict__)
    for name, value in name_args.items():
        if name in include_args:
            parts.append(f"{name}={value}")
    if parts:
        name = " ".join(parts)
    else:
        name = "default"
    return name


def prepare_repetition_directory(experiment_dir, args):
    num_existing = [int(x.name) for x in experiment_dir.iterdir() if x.is_dir()]
    if len(num_existing) == 0:
        index = 0
    else:
        index = max(num_existing)
        if not (experiment_dir / f"{index}" / f"metrics_best.json").exists():
            # last experiment wasn't finished
            shutil.rmtree(experiment_dir / f"{index}")
        else:
            if len(num_existing) == args.experiment_repetition:
                return None
            index += 1
    repetition_dir = experiment_dir / f"{index}"
    repetition_dir.mkdir(parents=True, exist_ok=True)
    if args.save_outputs:
        for sub in ["img_mel", "img_wavs", "wavs"]:
            (repetition_dir / sub).mkdir(parents=True, exist_ok=True)
    with open(str(repetition_dir / 'args.json'), 'w') as convert_file:
        json_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in dict(args.__dict__).items()}
        convert_file.write(json.dumps(json_args, indent=4))
    return repetition_dir


def prepare_experiment_directory(args):
    experiment_name = get_experiment_name(args)
    experiment_dir = Path(args.output_dir) / experiment_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir

###################################################################################################################
###################################################################################################################
