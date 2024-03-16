import os
import random
import sys
import time
from subprocess import check_output

import discord

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
GEN_SCRIPT = os.path.join(ROOT, "transformer", "run.py")

REPLY_THRES = 1200
CLARKE_THRES = 1800

OK = b"\xf0\x9f\x86\x97".decode("utf-8")
A = b"\xf0\x9f\x87\xa6".decode("utf-8")
B = b"\xf0\x9f\x87\xa7".decode("utf-8")
C = b"\xf0\x9f\x87\xa8".decode("utf-8")
CHECK = b"\xe2\x9c\x85".decode("utf-8")
PLUS = b"\xe2\x9e\x95".decode("utf-8")
MINUS = b"\xe2\x9e\x96".decode("utf-8")
TILDE = b"\xe3\x80\xb0\xef\xb8\x8f".decode("utf-8")
GRADES = (
    (A, PLUS),
    (A,),
    (A, MINUS),
    (B, PLUS),
    (B,),
    (B, MINUS),
    (C, PLUS),
    (C,),
    (C, MINUS),
    (TILDE),
    (CHECK, TILDE),
    (CHECK,),
    (CHECK, PLUS),
    (PLUS),
)

with open("quotes.txt", "r") as f:
    QUOTES = f.read().strip().split("\n\n")

intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
client = discord.Client(intents=intents)

last_reply_ok = 0
last_clarke = 0


@client.event
async def on_ready():
    print("I am ready.")


@client.event
async def on_message(message):
    global last_reply_ok, last_clarke

    if message.author == client.user:
        return
    if "clarke" in message.author.name:
        return

    content = message.content.lower()
    now = time.time()

    # React "OK"
    if random.randint(0, 25) == 0:
        await message.add_reaction(OK)

    # Give a grade.
    if random.randint(0, 35) == 0:
        grade = random.choice(GRADES)
        for letter in grade:
            await message.add_reaction(letter)

    # Quote.
    if ("clarke" in content and now - last_clarke > CLARKE_THRES) or \
            "sudo clarke" in content:
        await message.channel.send(random.choice(QUOTES))
        last_clarke = now

    # Reply OK if message contains OK.
    if "ok" in content:
        if now - last_reply_ok > REPLY_THRES:
            await message.reply("OK")
            last_reply_ok = now

    if "cnix" in content:
        await message.reply("Try cnix, the Linux distro that I built myself.\n"
                            "http://54.176.105.157:5555/")

    # Run the GAN
    words = content.strip().split()
    if len(words) == 2 and words[0] == "gen":
        # Start discord typing
        async with message.channel.typing():
            out = check_output([sys.executable, GEN_SCRIPT, words[1]], cwd=os.path.join(ROOT, "transformer"))
            out = out.decode("utf-8").strip()

        msg = "Generated text:\n"
        msg += out
        await message.reply(msg)


if __name__ == "__main__":
    with open("token.txt", "r") as f:
        token = f.read().strip()
    client.run(token)
