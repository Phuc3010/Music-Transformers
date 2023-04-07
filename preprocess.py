from tokenizer import *

if __name__ == "__main__":

    path = "./datasets/midi_songs/*.mid"
    midi_list = load_midi_folder(path)
    midi_frames, notes_frame, times_frame = midi2vocabulary(midi_list)
    # print(notes_frame)
    _time_tensor = vocabulary2tensor(times_frame, "time")
    _note_tensor = vocabulary2tensor(notes_frame, "note")

    torch.save(_time_tensor, "./data_gen/FPT_TIME_FRAMES.pt")
    torch.save(_note_tensor, "./data_gen/FPT_NOTE_FRAMES.pt")

    print("[+] Preprocessing successfully")
 