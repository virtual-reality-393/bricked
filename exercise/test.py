import pyaudio

py = pyaudio.PyAudio()

count = py.get_device_count()

for i in range(count):
    print(py.get_device_info_by_index(i))
    