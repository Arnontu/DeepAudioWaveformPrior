import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display

plt.style.use('seaborn-whitegrid')
plt.rc('font', family='serif', size=14)
mpl.rcParams['figure.dpi'] = 300


def annot_max(x, y, ax=None, offset=0, color='orange'):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    plt.scatter(xmax, ymax, 20, color=color)
    # plt.plot(xmax,ymax,'o',)
    text = "({:.3f},{:.3f})".format(xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="center", color=color)

    ax.annotate(text, xy=(xmax, ymax), xytext=(0.6, (offset + 1) * .1), **kw)
    # ax.annotate(text, xy=(xmax, ymax), xytext=(0.1+.3*offset, .1), **kw)

    return ax, xmax


def _plot_metric(metric, base_dir, show_estimates_vs_noise=True,save=True):
    iterations_i = np.arange(len(metric))
    reference = np.full(len(metric), metric.get_ref())
    fig = plt.figure()
    estimates_vs_clean = metric.get_clean()
    estimates_vs_noise = metric.get_noisy()
    plt.plot(iterations_i, estimates_vs_clean, label=f"{metric}(Estimates, Clean)", color='red')
    if show_estimates_vs_noise:
        plt.plot(iterations_i, estimates_vs_noise, label=f"{metric}(Estimates, Noisy)", color='green')
    ax, xmax = annot_max(iterations_i, estimates_vs_clean, color='red')
    ax.annotate("", xy=(0, reference[0]), xytext=(ax.get_xlim()[0], reference[0]),
                arrowprops=dict(arrowstyle="->"))
    plt.plot(reference, label=f"{metric}(Noisy, Clean)", color='orange')
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel(f"{metric}")
    plt.title(f"{metric} Comparison")
    locs, _ = plt.yticks()
    ##################
    ### to prevent target ytick overlap with other yticks
    ytickthresh = .25
    yticks = sorted(list(locs) + [reference[0]])
    ytickdiff = (locs[-1] - locs[0]) / (len(locs) - 1)
    tindex = yticks.index(reference[0])
    if tindex < len(yticks) - 1 and yticks[tindex + 1] - yticks[tindex] < ytickthresh * ytickdiff:
        yticks.pop(tindex + 1)
    elif tindex > 0 and yticks[tindex] - yticks[tindex - 1] < ytickthresh * ytickdiff:
        yticks.pop(tindex - 1)
    #################
    plt.yticks(np.array(yticks))
    if metric.ylim:
        plt.ylim(bottom=metric.ylim)
    plt.tight_layout()
    if save:
        plt.savefig(str(base_dir / f"{metric}.png"))
        plt.close(fig)
    else:
        plt.show()


def _plot_loss(basedir, losses):
    fig = plt.figure()
    plt.plot(losses)
    plt.title("Train Loss vs Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(str(basedir / f"loss.png"))
    plt.close(fig)


def _plot_wavs(wavs, sr, prefix=""):
    fig, ax = plt.subplots(nrows=len(wavs), ncols=1, sharex=True)
    titles = ["Noisy(Target)", "Output", "Clean Source"]
    for i, wav in enumerate(wavs):
        librosa.display.waveplot(wav.squeeze(), sr=sr, ax=ax[i])
        ax[i].set(title=titles[i])
        ax[i].label_outer()
    plt.subplots_adjust(hspace=0.5, wspace=.1)
    plt.savefig(prefix + '_waves.png')
    plt.close(fig)


def _plot_wav_to_stft(wavs, sr, n_fft=512, hop_length=256, prefix=""):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.subplots(nrows=len(wavs), ncols=1, sharex=True)
    titles = ["Noisy Target", "Output", "Clean Source"]
    for i, wav in enumerate(wavs):
        amp = np.abs(librosa.stft(wav.squeeze(), n_fft=n_fft, hop_length=hop_length))
        D = librosa.amplitude_to_db(amp, ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', x_axis='time',
                                       sr=sr, ax=ax[i])
        ax[i].set(title=titles[i])
        ax[i].label_outer()
    plt.subplots_adjust(hspace=0.2, wspace=.1)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.savefig(prefix + '_spectrograms.png')
    plt.close(fig)
# def _plot_after(base_dir, metric, show_estimates_vs_noise=True, nameprefix="after_"):
#     target = torchaudio.load(base_dir / "target.wav")[0]
#     gt = torchaudio.load(base_dir / "gt.wav")[0]
#     estimates_vs_gt = np.load(base_dir / f"{metric}_estimates_vs_gt.npy")
#     estimates_vs_target = np.load(base_dir / f"{metric}_estimates_vs_target.npy")
#     _plot_metric(estimates_vs_gt, estimates_vs_target, target, gt, metric, base_dir,
#                  show_estimates_vs_noise=show_estimates_vs_noise, nameprefix=nameprefix)
