import random
import sys
import string

from train import *

length = 200
if len(sys.argv) > 3:
    length = int(sys.argv[3])

model.load_state_dict(torch.load("models/"+sys.argv[1]+".pt", map_location=device))
prompt = sys.argv[2].strip().split() if len(sys.argv) > 2 else None
for word in gen_sample(length, prompt):
    if word == "i":
        word = "I"
    if word.strip().startswith("*"):
        print()
    if not word.startswith(tuple(string.punctuation)):
        print(" ", end="")
    print(word, end="", flush=True)
print()
