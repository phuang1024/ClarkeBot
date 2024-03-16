Make two directories: `texts` and `models`.

Create any `.txt` files in `texts`, holding the plaintext of the documents to
train on. They will automatically be tokenized.

Run `python train.py name` to train on `texts/name.txt` and save to
`models/name.pt`.

Adjust train parameters hardcoded in `train.py` as needed.

Run `./train_all.sh` as a shortcut to train (separate models) for all `.txt`
files in `texts`.
