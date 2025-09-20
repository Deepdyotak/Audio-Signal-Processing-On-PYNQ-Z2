# To record
from pynq.overlays.base import BaseOverlay
base = BaseOverlay("base.bit")
pAudio = base.audio
pAudio.set_volume(50)
pAudio.select_line_in()
pAudio.bypass(seconds=5)
pAudio.record(40)
pAudio.save("recording_1.wav")
#recording

# To play a saved audio 
import wave
import numpy as np
from pynq.overlays.base import BaseOverlay
base = BaseOverlay("base.bit")
pAudio = base.audio

pAudio.select_microphone()
pAudio.bypass(seconds=5)
pAudio.load("recording_1.wav")
pAudio.play()
