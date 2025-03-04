from RealtimeTTS import TextToAudioStream, SystemEngine, PiperEngine, PiperVoice
from RealtimeSTT import AudioToTextRecorder
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
from multiprocessing import Process, Queue
from multiprocessing.connection import PipeConnection

class VoiceAssistant:

    def __init__(self, language : str = "en", voice : str = "models/joe", in_audio : int = 11, out_audio : int = 13):
        # self.message_history = []

        # self.message_history.append({"role": "system","content": "You are trying to help a person stack lego bricks on top of each other. You will get extra information from a program that can detect and generate brick stack orders. Your job is to guide the user to stack the bricks in the order given by the program. Do not inform the user of anything that they didn't ask for"})
        # self.message_history.append({"role": "system", "content": "In front of the person there are currently, 1 red brick, 3 yellow bricks, 3 green bricks and 2 blue bricks"})
        # self.message_history.append({"role": "system", "content": "The current stack order is [red,green,blue,green,yellow,green,yellow,blue,yellow,blue]"})
        # print("LLM")
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
        
        # print("STT Recorder")
        # self.stt_recorder = AudioToTextRecorder(language=language,input_device_index=in_audio)


        self.language = language
        self.voice = voice
        self.in_audio = in_audio
        self.out_audio = out_audio
        self.message_queue = Queue()

        self.tts_process = Process(target=self.__tts_process__,args=(self.message_queue,))
        self.tts_process.start()


        
    def __tts_process__(self,queue : Queue):
        message = ""

        tts_voice = PiperVoice(model_file=self.voice + ".onnx", config_file=self.voice + ".txt")
        tts_engine = PiperEngine(voice = tts_voice,debug=True)
        tts_stream = TextToAudioStream(engine = tts_engine,output_device_index=self.out_audio,language=self.language)
        while True:
            message = queue.get()
            if message == "KILL_PROCESS":
                exit()
                
            tts_stream.feed(message)
            tts_stream.play()




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

    
    def play_message(self,message : str = "DEFAULT_MESSAGE") -> None:
        self.message_queue.put(message)

if __name__ == "__main__":
    print("Setting up assistant")
    assistant = VoiceAssistant()
    print("adding message")
    assistant.play_message("This is a test")
    print("adding message")
    assistant.play_message("This is a test")
    print("adding message")
    assistant.play_message("This is a test")
    print("adding message")
    assistant.play_message("This is a test")

    print("adding kill signal")
    assistant.play_message("KILL_PROCESS")

    print("waiting for process")
    assistant.tts_process.join()

