import pyaudio

p = pyaudio.PyAudio()
print("Available PyAudio Input Devices:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    # We only care about devices that can be used for input
    if info['maxInputChannels'] > 0:
        print(f"Index: {i}, Name: {info['name']}")
p.terminate()