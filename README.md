# PepGAN

PepGAN is activity-aware generative adversarial network for peptide sequence generation.

Details about PepGAN methodology can be found here

https://chemrxiv.org/articles/Generating_Ampicillin-Level_Antimicrobial_Peptides_with_Activity-Aware_Generative_Adversarial_Networks/12116136

PepGAN employs part of the benchmarking platform Texygen for more information see

https://github.com/geek-ai/Texygen

# Usage

* PepGAN demo run can be made using following command
  
  ```
  python main.py -g pepgan -t real -d data/amp.txt
  ```

* To run PepGAN on own sequences first you have to train activity predictor using own provided positive and negative samples. Links to the locations of both sets should be added into the `../PepGAN/activity_predictor/classifiertrain.py` file. Then activity predictor training can start
  
  ```
  python classifiertrain.py
  ```

  After that link to the trained model (.json and .h5 files) should be added into the `../PepGAN/models/Pepgan/PepganReward.py` file. Afterwards when running PepGAN link to the positive dataset which will be used for discriminator training also must be provided

  ```
  python main.py -g pepgan -t real -d <positive dataset location>
  ```
