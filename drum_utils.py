import pretty_midi
import numpy as np
import torch
import copy

def get_comp():
    
    standard = {36: 'bass', 38: 'snare', 
                46: 'open hi-hat', 42: 'closed hi-hat', 
                50: 'high tom', 47: 'mid tom', 43: 'low tom', 
                49: 'crash', 51: 'ride'}
    
    encoded = {'bass': 0, 'snare': 1, 
               'closed hi-hat': 2, 'open hi-hat': 3, 
               'high tom': 4, 'mid tom': 5, 'low tom': 6, 
               'crash': 7, 'ride': 8}
    
    return standard, encoded


def map_unique_drum(note):
    
    pitch = note.pitch
    standard, encoded = get_comp()
    
    # partial mapping
    map_to_standard = {36: 36, # bass 
                       37: 38, 38: 38, 40: 38,# snare
                       44: 42, 22: 42, 42: 42, # closed hi-hat
                       46: 46, 26: 46, # open hi-hat
                       50: 50, 48: 50, # high tom
                       43: 43, 58: 43, # low tom
                       47: 47, 45: 47, # mid tom
                       49: 49, 52: 49, 55: 49, 57: 49, # crash
                       51: 51, 53: 51, 59: 51} #ride
    
    if pitch not in standard.keys():
        if pitch in map_to_standard.keys():
            note.pitch = map_to_standard[pitch]
        else:
            return False
    
    return True

def change_fs(beats, target_beats=16):
        
    quarter_length = beats[1] - beats[0]
    changed_length = quarter_length /(target_beats / 4)
    changed_fs = 1 / changed_length
    
    return changed_fs

def quantize_drum(inst, fs, start_time, comp=9):
    
    fs_time = 1 / fs
    end_time = inst.get_end_time()
    
    standard, encoded = get_comp()
    
    quantize_time = np.arange(start_time, end_time + fs_time, fs_time)
    max_step = len(quantize_time)
    
    hit_roll = np.zeros((quantize_time.shape[0], comp))
    velocity_roll = np.zeros((quantize_time.shape[0], comp))
    offset_roll = np.zeros((quantize_time.shape[0], comp))
    
    step_notes = [[] for _ in range(max_step)]
    
    for note in inst.notes:
        if not map_unique_drum(note):
            continue
        
        start_index = np.argmin(np.abs(quantize_time - note.start))
        
        inst_index = encoded[standard[note.pitch]]
        step_notes[start_index].append(note)
    
    # multiple hits on the same drum at same time step -> keep max velocity
    for step in range(max_step):
        for drum in range(comp):
            drum_notes = [note for note in step_notes[step] if encoded[standard[note.pitch]] == drum]
            if len(drum_notes) > 1:
                note = max(drum_notes, key=lambda n: n.velocity)
            elif len(drum_notes) == 1:
                note = drum_notes[0]
            else:
                note = None

            if note:
                hit_roll[step, drum] = 1
                velocity_roll[step, drum] = note.velocity / 127.0
                offset_roll[step, drum] = (quantize_time[step] - note.start) / fs_time - 0.5
    
    return hit_roll, velocity_roll, offset_roll, max_step

def to_tensors(inst, fs, start_time, split_bars=2, hop_size=16, steps_per_bar=16, humanize = False, tapify = False, fixed_velocities = False):
    hit_vectors = []
    velocity_vectors = []
    offset_vectors = []

    hit_roll, velocity_roll, offset_roll, max_step = quantize_drum(inst, fs, start_time)

    total_bars = int(np.ceil(max_step / steps_per_bar))
    padded_length = steps_per_bar * total_bars

    hit_roll_padded = np.zeros((padded_length, hit_roll.shape[1]))
    velocity_roll_padded = np.zeros((padded_length, velocity_roll.shape[1]))
    offset_roll_padded = np.zeros((padded_length, offset_roll.shape[1]))

    hit_roll_padded[:len(hit_roll)] = hit_roll
    velocity_roll_padded[:len(velocity_roll)] = velocity_roll
    offset_roll_padded[:len(offset_roll)] = offset_roll

    hit_vectors.append(hit_roll_padded)
    velocity_vectors.append(velocity_roll_padded)
    offset_vectors.append(offset_roll_padded)

    if len(hit_vectors) == 0:
        return torch.zeros(0), torch.zeros(0), torch.zeros(0)

    hit_vectors = np.array(hit_vectors) # Shape : (1, max step, 9)
    velocity_vectors = np.array(velocity_vectors)
    offset_vectors = np.array(offset_vectors)
    
    hit_vectors = np.squeeze(hit_vectors, axis=0)  # Shape: (max step, 9)
    velocity_vectors = np.squeeze(velocity_vectors, axis=0)
    offset_vectors = np.squeeze(offset_vectors, axis=0)
    
    # input tensors for the encoder.
    in_hits = copy.deepcopy(hit_vectors)
    in_velocities = copy.deepcopy(velocity_vectors)
    in_offsets = copy.deepcopy(offset_vectors)
    
    if humanize:
        in_velocities[:] = 0
        in_offsets[:] = 0
        
    if tapify:
        argmaxes = np.argmax(in_velocities, axis=1)
        in_hits[:] = 0
        in_velocities[:] = 0
        in_offsets[:] = 0
        in_hits[:, 3] = hit_vectors[np.arange(padded_length), argmaxes] # 3 = open hi-hat
        in_velocities[:, 3] = velocity_vectors[np.arange(padded_length), argmaxes]
        in_offsets[:, 3] = offset_vectors[np.arange(padded_length), argmaxes]
        
    if fixed_velocities:
        in_velocities[:] = 0

    seqs = np.concatenate([hit_vectors, velocity_vectors, offset_vectors], axis=1) # Shape : (max step, 27)
    input_seqs = np.concatenate([in_hits, in_velocities, in_offsets], axis=1) # Shape : (max step, 27)
    
    seqs_tensor = torch.tensor(seqs, dtype=torch.float32)
    input_tensor = torch.tensor(input_seqs, dtype=torch.float32)

    if split_bars:
        window_size = steps_per_bar * split_bars # 32
        hop_size = window_size if hop_size is None else hop_size
        seqs_tensor = _extract_windows(seqs_tensor, window_size, hop_size) # (number of window?, window size = 32, 27)
        input_tensor = _extract_windows(input_tensor, window_size, hop_size)
        
        if len(seqs_tensor) == 0 or len(input_tensor) == 0:
            return None, None, None
        
        seqs_tensor = torch.stack(seqs_tensor) # (number of window?, window size = 32, 27)
        input_tensor = torch.stack(input_tensor)
        
    combined_tensor = torch.cat((input_tensor, seqs_tensor), dim=2) # (number of window?, window size = 32, 54)
        
    return seqs_tensor, input_tensor, combined_tensor

def _extract_windows(tensor, window_size, hop_size):
    if len(tensor) < window_size:
        return []  # 32 timestep 길이가 되지 않는 data가 있어서 그냥 pass 하기로 함
    return [tensor[i:i + window_size, :] for i in range(
        0, len(tensor) - window_size + 1, hop_size)]
    
def from_tensors_to_midi(tensor, steps_per_quarter=16, comp=9, fs=None, tempo_bpm=120, source_ppq=96):
    # Convert tensor to MIDI with proper resolution handling for Ableton Live exports.
    midi_data = pretty_midi.PrettyMIDI(resolution=source_ppq, initial_tempo=tempo_bpm)
    
    # Use the model's actual temporal resolution
    model_steps_per_quarter = 16  # What model was trained with
    
    # FIXED: Better step length calculation for proper tempo alignment
    # Convert model steps to MIDI ticks based on the PPQ resolution
    ticks_per_model_step = source_ppq / model_steps_per_quarter  # 96/16 = 6 ticks per step
    seconds_per_tick = (60.0 / tempo_bpm) / source_ppq
    step_length = ticks_per_model_step * seconds_per_tick
    
    # Create drum instrument
    drum_instrument = pretty_midi.Instrument(program=0, is_drum=True, name='Drums')
    midi_data.instruments.append(drum_instrument)
    standard, encoded = get_comp()
    
    for i in range(tensor.shape[0]):
        for j in range(comp):
            # FIXED: Better hit threshold to avoid noise
            hit = tensor[i, j].item()
            if hit > 0.15:  # Slightly higher threshold for cleaner output
                velocity = tensor[i, j + comp].item()
                offset = tensor[i, j + 2 * comp].item()
                
                # Convert to MIDI values with much better velocity scaling
                pitch = list(standard.keys())[j]
                
                # FIXED: More conservative velocity mapping
                velocity_base = velocity * 70 + 40  # 40-110 range (more realistic)
                velocity_noise = np.random.uniform(-3, 3)  # Subtle variation
                velocity_midi = int(velocity_base + velocity_noise)
                velocity_midi = max(velocity_midi, 40)  # Realistic minimum
                velocity_midi = min(velocity_midi, 120)  # Realistic maximum
                
                # FIXED: Much tighter offset scaling to prevent timing drift
                offset = offset * 0.08  # Even tighter timing
                
                # Calculate timing with proper resolution
                start_time = (i + offset) * step_length
                
                # FIXED: Better note duration based on tempo and drum type
                if j in [0, 1]:  # Bass, snare - longer sustain
                    duration = step_length * 0.8
                elif j in [2, 3]:  # Hi-hats - shorter
                    duration = step_length * 0.3
                else:  # Toms, cymbals - medium
                    duration = step_length * 0.6
                
                duration = max(duration, 0.05)  # Minimum duration
                end_time = start_time + duration
                
                # Create note
                note = pretty_midi.Note(
                    velocity=velocity_midi,
                    pitch=pitch,
                    start=start_time,
                    end=end_time
                )
                drum_instrument.notes.append(note)
    
    midi_data.instruments.append(drum_instrument)
    
    return midi_data