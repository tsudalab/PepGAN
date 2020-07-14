# PepGAN

To run PepGAN on antimicrobial peptides use following command:

python main.py -g pepgan -t real -d data/output_amp.txt

To run PepGAN on own sequences you have to train activity predictor first using own data via following command:

python classifiertrain.py



PepGAN employs part of the benchmarking platform Texygen for more details see

https://github.com/geek-ai/Texygen
