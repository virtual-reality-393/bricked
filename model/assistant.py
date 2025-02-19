from RealtimeTTS import TextToAudioStream, SystemEngine, PiperEngine, PiperVoice
from RealtimeSTT import AudioToTextRecorder
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer

class VoiceAssistant:

    def __init__(self, language : str = "en", voice : str = "models/joe", in_audio : int = 11, out_audio : int = 13):
        self.message_history = [{"role": "system","content": "You are a personal voice assistant controlled via a pair of smart glasses. You are also very moody most of the time. Please keep your messages concise"}]
        print("LLM")
        # self.llm_model = AutoModelForCausalLM.from_pretrained(
        #     "jpacifico/Chocolatine-3B-Instruct-DPO-Revised",
        #     device_map="cuda",
        #     torch_dtype="auto",
        #     trust_remote_code=True,
        # )
        # self.llm_tokenizer = AutoTokenizer.from_pretrained("jpacifico/Chocolatine-3B-Instruct-DPO-Revised")

        # self.pipeline  = pipeline(
        #     "text-generation",
        #     model=self.llm_model,
        #     tokenizer=self.llm_tokenizer,
        # )
        print("TTS Voice")
        self.tts_voice = PiperVoice(model_file=voice + ".onnx", config_file=voice + ".txt")
        print("TTS Engine")
        self.tts_engine = PiperEngine(voice = self.tts_voice,debug=True)
        print("TTS Stream")
        self.tts_stream = TextToAudioStream(engine = self.tts_engine,output_device_index=out_audio,language=language)
        print("STT Recorder")
        self.stt_recorder = AudioToTextRecorder(language=language,input_device_index=in_audio)



    def generate_response(self,prompt : str, max_new_tokens : int = 500, temperature : float = 0.5) -> str:
    
        self.message_history.append({"role": "user", "content": prompt})

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            "temperature": temperature,
            "do_sample": False,
        }

        output = self.pipeline(self.message_history, **generation_args)

        response = output[0]['generated_text']
    
        self.message_history.append({"role":"assistant", "content":response})

        return response
    
    def get_message(self) -> str:
        return self.stt_recorder.text()

    
    def play_message(self,message : str = "DEFAULT_MESSAGE", play_async : bool = False) -> None:
        print("Feeding message")
        self.tts_stream.feed(message)
        if play_async:
            print("Playing ASYNC")
            self.tts_stream.play_async()
        else:
            print("Playing SYNC")
            self.tts_stream.play()

if __name__ == "__main__":
    print("Setting up assistant")
    assistant = VoiceAssistant()
    print("Assistant is ready")
    print("Playing message")
    assistant.play_message("Testing, testing, 1, 2, 3, is it working?")