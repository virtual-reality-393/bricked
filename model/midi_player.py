from time import sleep

import rtmidi
import time
import rtmidi.midiconstants as midiconstants
import threading


class MIDIPlayer:
    def __init__(self, midi_port = "MIDI_OUT"):
        self.midiout = rtmidi.MidiOut()
        ports = self.midiout.get_ports()
        self.midiout.open_port(2)


    def play_note(self,key,velocity,note_time = 1):
        self.midiout.send_message([midiconstants.NOTE_ON,key,velocity])

        time.sleep(note_time)

        self.midiout.send_message([midiconstants.NOTE_OFF,key,0])

    def play_note_async(self,key,velocity,note_time = 1):
        threading.Thread(target=self.play_note,args=(key,velocity,note_time)).start()



def main():
    player = MIDIPlayer()
    time.sleep(3)
    print("Playing")
    player.play_note_async(75,127,1)
    player.play_note_async(76,127,1)
    player.play_note_async(77,127,1)
    print("Finished Playing")
    while True:
        time.sleep(0.1)



