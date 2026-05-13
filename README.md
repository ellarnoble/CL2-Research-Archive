# CL2-Research-Archive
# Project Overview
This study implements a model for multilingual grapheme-to-phoneme (G2P) mapping across eight Cyrillic/Latin languages: Bulgarian, English, Indonesian, Macedonian, Russian, Spanish, Tagalog and Ukrainian. The model uses an encoder-decoder architecture and greedy decoding for autoregressive phoneme generation.  

Data is taken from SIGMORPHON Shared Task (2024) which contains training, validation and test files for each language. This can be found at: 
[SIGMORPHON (2024)](https://github.com/sigmorphon/2024G2PST)

Each file contains paired orthographic and phonemic transcriptions of words; these are prepended with a script and language token ID prior to being processed by the model. 

# Structure of Repository
The repository contains the following files: 
- script.py: contains all preprocessing, modelling, training and evaluation code for the model. This can be run from top to bottom.
- results.txt: contains the execution logs from running the script.py file. This includes per-epoch training statistics and test results.
- requirements.txt: contains all dependencies required to reproduce the environment for running script.py.
- slurm.sh: contains the slurm job submission script for running script.py on the University of Manchester CSF.

Running script.py will also generate the following files: training_curves.png; best_model.pt and lang_embedding_similarity.png 

# Running the Code
All code was executed on the University of Manchester CSF using a single A40 40GB GPU with 4 CPU cores and 32GB memory, via the slurm.sh file inside a Python 3.13.1 virtual environment. To reproduce the environment, create a new virtual environment and install dependencies using pip install -r requirements.txt. To run the script, the SIGMORPHON data repository (2024G2PST) must be cloned into the same directory as script.py.

Total run time was approximately 2.75 hours. The experiment's results were tested for reproducibility by re-running the final script; subsequent runs yielded identical results to four decimal places.
