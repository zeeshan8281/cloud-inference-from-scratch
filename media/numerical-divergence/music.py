"""Original quiet electronic bed; no samples or copied reference audio."""
import math
import struct
import wave
from pathlib import Path

rate = 44100
seconds = 16
notes = (110.0, 164.8138, 220.0, 261.6256)
out = bytearray()
for index in range(rate * seconds):
    t = index / rate
    envelope = min(1, t/1.2, (seconds-t)/1.5)
    pad = sum(math.sin(2*math.pi*n*t + .13*math.sin(t*.8)) for n in notes)/len(notes)
    pulse = 0.0
    for start, frequency in ((.8,880), (2.5,1046.5), (5,1318.5), (8,1046.5), (11,880)):
        age = t-start
        if 0 <= age < .7:
            pulse += math.sin(2*math.pi*frequency*age)*math.exp(-age*9)*min(1,age/.015)
    value = max(-1,min(1,envelope*(pad*.09+pulse*.025)))
    out.extend(struct.pack('<h',round(value*32767)))
with wave.open(str(Path(__file__).with_name('music.wav')), 'wb') as wav:
    wav.setparams((1,2,rate,0,'NONE','not compressed'))
    wav.writeframes(out)
assert len(out) == rate * seconds * 2
