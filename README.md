# PepGAN

PepGAN is activity aware generative adversarial network for peptide sequence generation.

Details about PepGAN methodology can be found here

https://chemrxiv.org/articles/Generating_Ampicillin-Level_Antimicrobial_Peptides_with_Activity-Aware_Generative_Adversarial_Networks/12116136

PepGAN employs part of the benchmarking platform Texygen for more information see

https://github.com/geek-ai/Texygen

# Usage

* To run PepGAN use following command:
  
  ```
  python main.py -g pepgan -t real -d data/output_amp.txt
  ```

* To run PepGAN on your own sequences you have to train activity predictor first using own data (positive and negative samples) via following command:
  
  ```
  cd activity_predictor
  python classifiertrain.py
  ```

  Afterwards link to the trained model should be added into the `../PepGAN/models/Pepgan/PepganReward.py` file <br /><br />

