# PepGAN

PepGAN is activity aware generative adversarial network for peptide sequence generation.

Details about PepGAN methodology can be found here

https://chemrxiv.org/articles/Generating_Ampicillin-Level_Antimicrobial_Peptides_with_Activity-Aware_Generative_Adversarial_Networks/12116136

PepGAN employs part of the benchmarking platform Texygen for more information see

https://github.com/geek-ai/Texygen

# Usage

* To run PepGAN use the following command
  
  ```
  python main.py -g pepgan -t real -d data/amp.txt
  ```

* To run PepGAN on own sequences first you have to train activity predictor using own provided positive and negative samples via
  
  ```
  cd activity_predictor
  python classifiertrain.py
  ```

  Then link to the trained model should be added into the `../PepGAN/models/Pepgan/PepganReward.py` file. Afterwards when running PepGAN link to dataset which will  be used for minimax game should be also provided

  ```
  python main.py -g pepgan -t real -d <your data base location>
  ```
