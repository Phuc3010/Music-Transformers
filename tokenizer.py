import torch
import glob
from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct
import numpy as np
import matplotlib.pyplot as plt


NUM_NOTES = 19
NUM_TIME = 4
SCALE_NOTE = 48
NOTE_SEQ = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23, 24, 26, 28, 29, 31]
NOTE_SEQ = [f"note_{i}" for i in NOTE_SEQ]
SYMBOL_SEQ = [f"symbol_{1<<i}" for i in range(NUM_TIME)]
TIME_SEQ = [f"note_{1<<i}" for i in range(NUM_TIME)]
SUPPORT_SEQ = ["<pad>", "<start>", "<end>"]
VOCABULARY = {
    "NOTE": SUPPORT_SEQ + NOTE_SEQ,
    "TIME": SUPPORT_SEQ + SYMBOL_SEQ  + TIME_SEQ
}

def load_midi_folder(path: str) -> list:
    list_path = glob.glob(path)
    midi_list = [mid_parser.MidiFile(_path) for _path in list_path]
    midi_list = list(filter(lambda midi: len(midi.time_signature_changes)==1 , midi_list))
    midi_list = list(filter(lambda midi: midi.time_signature_changes[0].numerator==2 and midi.time_signature_changes[0].denominator==4 , midi_list))
    print(f"[INFO] Load successfully {len(midi_list)} file midi from source")
    return midi_list

def plot_midi_roll(midi: list) -> None:
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.set_facecolor('White')
    start = np.zeros(NUM_NOTES)
    label = np.arange(NUM_NOTES)
    scales = list(filter(lambda word: "symbol" not in word, midi))
    scales = [int(i.split("_")[1])//12 for i in scales]
    min_scale = 12*min(scales)
    for word in midi:
        if "symbol" in word:
            start += 32//int(word.split("_")[-1])
        else:
            pitch = int(word.split("_")[1])
            pitch -= min_scale
            duration = 32//int(word.split("_")[-1])
            width = np.zeros(NUM_NOTES)
            width[NOTE_SEQ.index(f"note_{pitch}")] = duration
            ax.barh(label, width, left=start, height=0.4, label=pitch, color="green")
            start += (duration+2)
    plt.xlabel("Time by tick")
    plt.ylabel("Pitch")
    plt.show()

def get_duration(time: int, ticks_per_beat: int) -> int:
    all_tick = 2*ticks_per_beat
    duration = (1<<4)
    diff = all_tick
    for current_pow in range(5):
        if diff > abs(all_tick//(1<<current_pow) - time):
            duration = (1<<current_pow)
            diff = abs(all_tick//(1<<current_pow) - time)
    return duration

def time2duration(time: int, ticks_per_beat: int) -> list:
    all_tick = 2*ticks_per_beat
    duration_list = []
    for current_pow in range(5):
        while time >= all_tick//(1<<current_pow):
            duration_list.append(1<<current_pow)
            time -= all_tick//(1<<current_pow)
    return duration_list

def midi2seq(midi_list: list) -> list:
    sequences = []
    tmp_cache = {
        "note": 0,
        "track": 0,
        "symbol": 0
    }
    for midi in midi_list:
        notes = midi.instruments[0].notes
        current_time = 0
        sequence = []
        tmp_cache["track"] += (notes[-1].end//(2*midi.ticks_per_beat) + (notes[-1].end%(2*midi.ticks_per_beat) != 0))
        sub_pitch = 0
        if len(midi.key_signature_changes):
            sub_pitch = midi.key_signature_changes[0].key_number
        for note in notes:
            diff = note.start - current_time
            note.pitch -= sub_pitch
            if diff > 0:
                seq_duration = time2duration(diff, midi.ticks_per_beat)
                seq_duration = [f"symbol_{i}" for i in seq_duration]
                current_time = note.start
                sequence += seq_duration
                tmp_cache["symbol"] += len(seq_duration)
            note_duration = get_duration(note.end-note.start, midi.ticks_per_beat)
            sequence += [f"note_{note.pitch}_{note_duration}"]
            duration = 2*midi.ticks_per_beat//(note_duration)
            current_time += duration
            tmp_cache["note"] += 1
        sequences.append(sequence)
    print("[INFO] Convert midi to sequences successfully")
    print(f"\t[+] Note: {tmp_cache['note']}")
    print(f"\t[+] Symbol: {tmp_cache['symbol']}")
    print(f"\t[+] Track: {tmp_cache['track']}")
    return sequences

# def seq2vocab_idx(sequences: list) -> list:
#     note_idx = []
#     time_idx = []
#     for sequence in sequences:
#         time_id = []
#         note_id = []
#         for word in sequence:
#             if "symbol" in word:
#                 time_id.append(VOCABULARY["TIME"].index(word))
#             else:
#                 word = word.split("_")
#                 note_id.append(VOCABULARY["NOTE"].index(f"note_{word[1]}"))
#                 time_id.append(VOCABULARY["TIME"].index(f"note_{word[2]}"))
#         note_idx.append(note_id)
#         time_idx.append(time_id)
#     return note_idx, time_idx

def scale_time(time_sequence: list) -> list:
    new_seq = []
    back_note = ""
    back_symbol = ""
    for word in time_sequence:
        val = int(word.split("_")[-1])
        if val == 16:
            if "note_" in word and back_note == "":
                new_seq.append("note_8")
                back_note = word 
            if "note_" in word and back_note == word:
                back_note = ""
                continue
            if "symbol_" in word and back_symbol == "":
                new_seq.append("symbol_8")
                back_symbol = word
            if "symbol_" in word and back_symbol == word:
                back_symbol = ""
                continue
        else:
            new_seq.append(word)
    return new_seq

def seq2seq_idx(sequences: list) -> list:
    time_sequences = []
    note_sequences = []
    
    # Split into time frame and note frame
    for midi_seq in sequences:
        note_seq = list(filter(lambda word: "note" in word, midi_seq))
        note_seq = list(map(lambda word: f"note_{word.split('_')[1]}", note_seq))
        time_seq = list(map(lambda word: f"{word.split('_')[0]}_{word.split('_')[-1]}" , midi_seq))
        time_seq = scale_time(time_seq)
        time_seq = list(map(lambda word: VOCABULARY["TIME"].index(word), time_seq))
        note_sequences.append(note_seq)
        time_sequences.append(time_seq)
    
    # Convert note frame to tone C
    for idx_seq, note_seq in enumerate(note_sequences):
        scales = [int(word.split("_")[-1])//12 for word in note_seq]
        min_scales = min(scales)*12
        note_seq = list(map(lambda word: f"note_{int(word.split('_')[-1])-min_scales-(int(word.split('_')[-1])%12==1 or int(word.split('_')[-1])%12==6)}", note_seq))
        note_seq = list(map(lambda word: VOCABULARY["NOTE"].index(word), note_seq))
        note_sequences[idx_seq] = note_seq

    return note_sequences, time_sequences


def note2tensor(note_idx: list, seq_len: int, step: int) -> torch.Tensor:
    tokens = torch.Tensor()
    for idx, note in enumerate(note_idx):
        for st in range(0,len(note),step):
            en = seq_len + st
            if en > len(note):
                break
            seq = torch.tensor([1]+note[st:en]+[2])
            tokens = torch.cat((tokens, seq.view(1, len(seq))), dim=0)
    tokens = tokens[torch.randperm(tokens.size(0))]
    return tokens

def split_track(time_list: int) -> list:
    time = 0.0
    tracks = []
    track = []
    for word in time_list:
        _word = VOCABULARY["TIME"][word]
        _time = 1/int(_word.split("_")[-1])
        time += _time
        track.append(word)
        if time == int(time):
            if len(track):
                tracks.append(track)
            track = []
            time = 0.0
    tracks.append(track)
    return tracks


def time2tensor(time_idx: list, track_len: int, step: int) -> torch.Tensor:
    sequences = []
    max_len_seq = 0
    for time in time_idx:
        track = split_track(time)
        for st in range(0, len(track), step):
            en = st + track_len
            if en > len(track):
                break
            seq = []
            for _track in track[st:en]:
                seq += _track
            if len(seq) > max_len_seq:
                max_len_seq = len(seq)
            sequences.append(seq)
    tokens = torch.Tensor()
    for seq in sequences:
        padding = [0 for _ in range(max_len_seq-len(seq))]
        seq = [1] + seq + [2] + padding
        tokens = torch.cat((tokens, torch.tensor(seq).view(1, len(seq))), dim=0)
    tokens = tokens[torch.randperm(tokens.size(0))]
    return tokens

def num2bin(num: int) -> list:
    seq = []
    cur = 0
    while num:
        while num >= 1/(1<<cur):
            num -= 1/(1<<cur)
            seq.append(1<<cur)
        cur += 1
    return seq


def process_timeframe(time_frame: torch.Tensor) -> list:
    time_list = []
    current_time = 0.0
    for symbol in time_frame:
        if symbol >= 0 and symbol <= 2:
            continue
        symbol = VOCABULARY["TIME"][symbol]
        value = 1/int(symbol.split("_")[-1])
        if current_time + value > 1.0:
            diff = 1.0 - current_time
            time_list += [f"{symbol.split('_')[0]}_{i}" for i in num2bin(diff)]
            current_time = 0.0
        elif current_time + value == 1.0:
            current_time = 0
            time_list.append(symbol)
        else:
            time_list.append(symbol)
            current_time += value
    return time_list


def process_noteframe(note_frame: torch.Tensor) -> list:
    note_list = []
    for idx_note in note_frame:
        if idx_note <= 2:
            continue
        value = VOCABULARY["NOTE"][idx_note].split("_")[-1]
        note_list.append(int(value))
    return note_list


def frame2midi(note_frame: torch.Tensor, time_frame: list, save_path: str=None, ticks_per_beat: int = 480, tempo: int=120, numerator: int=2, denominator: int=4) -> mid_parser.MidiFile:
    mido_obj = mid_parser.MidiFile()
    mido_obj.ticks_per_beat = ticks_per_beat
    mido_obj.time_signature_changes.append(ct.TimeSignature(numerator=numerator, denominator=denominator, time=0))
    mido_obj.key_signature_changes.append(ct.KeySignature(key_name="C", time=0))
    mido_obj.tempo_changes.append(ct.TempoChange(tempo=tempo, time=0))
    
    track = ct.Instrument(program=0, is_drum=False, name="piano")
    mido_obj.instruments = [track]

    length_frame = len(time_frame)
    length_note = len(note_frame)
    
    current_time = 0
    idx_note = 0
    
    current_note = {
        "start": 0,
        "end": 0,
        "pitch": 0,
        "velocity": 90
    }
    
    for idx_time, time in enumerate(time_frame):
        if time == "symbol_16":
            continue
        value = 1/int(time.split("_")[-1])
        current_time += int(numerator*ticks_per_beat*value)
        # if "symbol" in time:
        #     current_note["start"] = current_time
        # else:
        current_note["end"] = current_time
        current_note["pitch"] = int(int(note_frame[idx_note]) + SCALE_NOTE)
        idx_note = (idx_note+1)%length_note
        note = ct.Note(**current_note)
        mido_obj.instruments[0].notes.append(note)
        current_note["start"] = current_time
    if save_path is not None:
        mido_obj.dump(save_path)
        print("[+] Save midi file sucessfully! File name:", save_path)
    return mido_obj


if __name__ == "__main__":
    path = "./datasets/midi_songs/*.mid"
    midi_list = load_midi_folder(path)
    sequences = midi2seq(midi_list)
    note_sequences, time_sequences = seq2seq_idx(sequences)

    note_tensor = note2tensor(note_sequences, 40, 40)
    time_tensor = time2tensor(time_sequences, 20, 20)
    torch.save(note_tensor, "./data_gen/note_tensor_v3.pt")
    torch.save(time_tensor, "./data_gen/time_tensor_v3.pt")

    print(time_tensor.size(), note_tensor.size())
    plot_midi_roll(sequences[1])