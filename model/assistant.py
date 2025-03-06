from RealtimeTTS import TextToAudioStream, SystemEngine, PiperEngine, PiperVoice
from RealtimeSTT import AudioToTextRecorder
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
from multiprocessing import Process, Queue
from multiprocessing.connection import PipeConnection

class VoiceAssistant:

    def __init__(self, language : str = "en", voice : str = "models/talesyntese", in_audio : int = 12, out_audio : int = 14):
        self.message_history = []
        self.message_history.append({"role": "system", "content": "Du er bindeedet mellem en bruger og et system, du skal oversætte brugerens input til den kommando du mener er tættest på hvad brugeren vil have, hvis dit output ikke matcher hvad systemet forventer vil det blive henkastet. Kommandoerne er alle i formatet {COMMAND}, og den inderste tekst betyder hvilken kommando der bliver sendt. Du skal så baseret på brugerens input sende det kommando signalet som er tættest på. Kommando 1: {NUM_BRICK} - Kan informere brugeren om hvor mange klodser der er | Kommando 2: {CURR_STACK} - Kan informere brugeren om den stabel af klodser de har samlet | Kommando 3: {BUILD_ORDER} - Kan informere brugeren om den rækkefølge det er meningen de skal samle. Du skal kun sende 1 kommando per besked du modtager"})

        print("LLM")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            "jpacifico/Chocolatine-3B-Instruct-DPO-Revised",
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained("jpacifico/Chocolatine-3B-Instruct-DPO-Revised")

        self.pipeline  = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
        )
        
        # print("STT Recorder")

    
        # self.stt_recorder = AudioToTextRecorder(language=language,input_device_index=in_audio,model="medium")


        # self.language = language
        # self.voice = voice
        # self.in_audio = in_audio
        # self.out_audio = out_audio
        # self.message_queue = Queue()

        # self.tts_process = Process(target=__tts_process__,args=(self.message_queue,))
        # self.tts_process.start()


        

    def __stt_process__(self,queue: Queue):
        pass



    def generate_response(self,prompt : str, max_new_tokens : int = 500, temperature : float = 0.4) -> str:
    
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

def __tts_process__(queue : Queue, voice = "models/joe", out_audio = 14, language = "da"):
        message = ""

        tts_voice = PiperVoice(model_file=voice + ".onnx", config_file=voice + ".txt")
        tts_engine = PiperEngine(voice = tts_voice,debug=True)
        tts_stream = TextToAudioStream(engine = tts_engine,output_device_index=out_audio,language=language)
        while True:
            message = queue.get()
            if message == "KILL_PROCESS":
                exit()

            tts_stream.feed(message)
            tts_stream.play()

if __name__ == "__main__":
    print("Setting up assistant")
    assistant = VoiceAssistant(language="da")
    while True:
        user_input = assistant.get_message()
        command = assistant.generate_response(user_input)

        if "NUM_BRICK" in command:
            assistant.play_message("Der er 5 klodser foran dig")
        elif "BUILD_ORDER" in command:
            assistant.play_message("Rækkefølgen er grøn, blå, rød, gul")
        elif "CURR_STACK" in command:
            assistant.play_message("Du har sat blå oven på en grøn, den næste du skal placere er en rød")
        else:
            assistant.play_message(command)
