#!/bin/bash
if [[ $# -eq 0 ]]
then
SOURCE= #<path_to_directory_with_audios>
else
SOURCE=$1
fi

python3 denoise.py \
    --experiment_repetition 20 \
    --show_every 250 \
    --noise_class GAUSSIAN \
    --noise_std 0.2 \
    --snr 2.5 \
    --depth 2 \
    --lstm_layers 4 \
    --source $SOURCE \
    --samplerate 16000 \
    --clip_length 2 \
