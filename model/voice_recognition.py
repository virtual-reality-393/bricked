from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

engine = SystemEngine(voice = "zira") # replace with your TTS engine
engine.set_voice(engine.get_voices()[1].name)

stream = TextToAudioStream(engine,language="da")
print("Playing")
stream.feed("Hvad kalder man en død bi? - En Zom-bi")
stream.play_async()
print("Finished")