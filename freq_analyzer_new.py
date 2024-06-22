# #### Time lapse improvement


# from flask import Flask, request, jsonify, render_template
# import os
# import librosa
# import numpy as np
# import warnings
# import soundfile as sf
# from scipy import signal
# import logging
# import noisereduce as nr
# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean

# warnings.filterwarnings('ignore')

# app = Flask(__name__)

# # Setup logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# def load_audio(audio_path, duration=None):
#     logging.debug(f"Loading audio file {audio_path} for {duration} seconds." if duration else f"Loading full audio file {audio_path}.")
#     audio, sr = librosa.load(audio_path, duration=duration)
#     return audio, sr

# def apply_band_pass_filter(segment, sr, lowcut, highcut):
#     logging.debug(f"Applying band-pass filter with lowcut {lowcut} Hz & highcut {highcut} Hz.")
#     sos = signal.butter(10, [lowcut, highcut], 'bp', fs=sr, output='sos')
#     filtered_segment = signal.sosfilt(sos, segment)
#     return filtered_segment

# def compare_audios(audio1_path, audio2_path, amp_tolerance_percent=20, freq_tolerance_percent=20, segment_duration=2.0, overlap=0.5):
#     logging.debug("Comparing audio files with segment overlap.")
#     audio1, sr1 = load_audio(audio1_path)
#     audio2, sr2 = load_audio(audio2_path)
    
#     if sr1 != sr2:
#         logging.error("Sample rates of the audio files do not match.")
#         return "Sample Rate Mismatch", None, None

#     segment1_path = "master_segment.wav"
#     segment2_path = "sample_segment.wav"

#     matched_segments = []
    
#     for start in np.arange(0, len(audio1) - int(segment_duration * sr1), int(segment_duration * sr1 * (1 - overlap))):
#         print("In first for loop")
#         segment1 = audio1[int(start):int(start + segment_duration * sr1)]
#         segment2_start_indices = np.arange(0, len(audio2) - int(segment_duration * sr2), int(segment_duration * sr2 * (1 - overlap)))
#         for segment2_start in segment2_start_indices:
#             print("In second for loop")
#             segment2 = audio2[int(segment2_start):int(segment2_start + segment_duration * sr2)]
            
#             max_amplitude1 = np.max(np.abs(segment1))
#             max_amplitude2 = np.max(np.abs(segment2))
#             amplitude_difference = abs(max_amplitude1 - max_amplitude2)
#             amplitude_tolerance = amp_tolerance_percent / 100.0 * abs(max_amplitude1)
            
#             dominant_freq1 = get_dominant_frequency(segment1, sr1)
#             dominant_freq2 = get_dominant_frequency(segment2, sr2)
#             frequency_difference = abs(dominant_freq1 - dominant_freq2)
#             frequency_tolerance = freq_tolerance_percent / 100.0 * abs(dominant_freq1)
            
#             logging.debug(f"Segment Comparison - Max Amplitude1: {max_amplitude1}, Max Amplitude2: {max_amplitude2}")
#             logging.debug(f"Segment Comparison - Amplitude Difference: {amplitude_difference}, Amplitude Tolerance: {amplitude_tolerance}")
#             logging.debug(f"Segment Comparison - Frequency Difference: {frequency_difference}, Frequency Tolerance: {frequency_tolerance}")
#             logging.debug(f"Segment Comparison - Dominant Frequency1: {dominant_freq1}, Dominant Frequency2: {dominant_freq2}")

#             if np.round(amplitude_difference) <= np.round(amplitude_tolerance) and np.round(frequency_difference) <= np.round(frequency_tolerance):
#                 distance, path = fastdtw(segment1, segment2, dist=euclidean)
#                 logging.debug(f"Segment Comparison - DTW Distance: {distance}")
#                 if distance < (amplitude_tolerance + frequency_tolerance):
#                     matched_segments.append((segment1, segment2))
#                     sf.write(segment1_path, segment1, sr1)
#                     sf.write(segment2_path, segment2, sr2)
#                     return "Match", segment1_path, segment2_path

#     if not matched_segments:
#         return "No Match", None, None

#     return "Partial Match", segment1_path, segment2_path

# def get_dominant_frequency(segment, sr):
#     logging.debug("Calculating dominant frequency.")
#     fft = np.fft.fft(segment)
#     freqs = np.fft.fftfreq(len(segment), 1/sr)
#     magnitude = np.abs(fft)
#     dominant_freq = freqs[np.argmax(magnitude)]
#     return abs(dominant_freq)

# def save_audio(audio, filepath):
#     with open(filepath, 'wb') as f:
#         f.write(audio.read())

# @app.route('/compare', methods=['POST'])
# def compare():
#     if 'audio' not in request.files:
#         logging.error("No audio file provided.")
#         return jsonify({'result': 'No audio file provided'}), 400

#     audio = request.files['audio']
#     audio_type = request.form['type']

#     if audio_type == 'master':
#         audio_path = r"master.wav"
#     elif audio_type == 'sample':
#         audio_path = r"sample.wav"
#     else:
#         logging.error("Invalid audio type provided.")
#         return jsonify({'result': 'Invalid audio type'}), 400

#     save_audio(audio, audio_path)

#     master_path = r"master.wav"
#     sample_path = r"sample.wav"
    
#     if os.path.exists(master_path) and os.path.exists(sample_path):
#         result, segment1_path, segment2_path = compare_audios(master_path, sample_path)
#         logging.debug(f"Comparison result: {result}")
#         response = {'result': result}
#         if segment1_path and segment2_path:
#             response['master_segment'] = segment1_path
#             response['sample_segment'] = segment2_path
#         return jsonify(response)

#     return jsonify({'result': 'Audio file saved'}), 200

# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
import os
import librosa
import numpy as np
import warnings
import soundfile as sf
from scipy import signal
import logging
import noisereduce as nr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

def load_audio(audio_path, duration=None):
    logging.debug(f"Loading audio file {audio_path} for {duration} seconds." if duration else f"Loading full audio file {audio_path}.")
    audio, sr = librosa.load(audio_path, sr=None, duration=duration)  # Ensure loading at the original sample rate
    logging.debug(f"Loaded audio file {audio_path} with {len(audio)} samples at {sr} Hz")
    return audio, sr

def apply_band_pass_filter(segment, sr, lowcut, highcut):
    logging.debug(f"Applying band-pass filter with lowcut {lowcut} Hz & highcut {highcut} Hz.")
    sos = signal.butter(10, [lowcut, highcut], 'bp', fs=sr, output='sos')
    filtered_segment = signal.sosfilt(sos, segment)
    return filtered_segment

def flatten(something):
    if isinstance(something, (list, tuple, set, range)):
        for sub in something:
            yield from flatten(sub)
    else:
        yield something

def compare_audios(audio1_path, audio2_path, amp_tolerance_percent=25, freq_tolerance_percent=25, segment_duration=2.0, overlap=0.5):
    logging.debug("Comparing audio files with segment overlap.")
    audio1, sr1 = load_audio(audio1_path)
    audio2, sr2 = load_audio(audio2_path)
    
    if sr1 != sr2:
        logging.error("Sample rates of the audio files do not match.")
        return "Sample Rate Mismatch", None, None

    segment1_path = "master_segment.wav"
    segment2_path = "sample_segment.wav"

    matched_segments = []

    # Ensure the segment duration is within the audio length
    max_duration = min(len(audio1), len(audio2)) / sr1
    segment_duration = min(segment_duration, max_duration - 0.01)  # Subtract a small value to ensure segment duration is less than max_duration

    segment_length = int(segment_duration * sr1)
    overlap_length = int(segment_length * overlap)

    logging.debug(f"Adjusted Segment length: {segment_length}, Overlap length: {overlap_length}")
    
    if segment_length >= len(audio1) or segment_length >= len(audio2):
        logging.error("Segment length is greater than or equal to the length of one of the audio files.")
        return "Segment length too large", None, None
    
    for start1 in range(0, len(audio1) - segment_length + 1, overlap_length):
        logging.debug(f"Segment 1 start index: {start1}")
        segment1 = audio1[start1:start1 + segment_length]
        for start2 in range(0, len(audio2) - segment_length + 1, overlap_length):
            logging.debug(f"Segment 2 start index: {start2}")
            segment2 = audio2[start2:start2 + segment_length]
            
            max_amplitude1 = np.max(np.abs(segment1))
            max_amplitude2 = np.max(np.abs(segment2))
            amplitude_difference = abs(max_amplitude1 - max_amplitude2)
            amplitude_tolerance = amp_tolerance_percent / 100.0 * abs(max_amplitude1)
            
            dominant_freq1 = get_dominant_frequency(segment1, sr1)
            dominant_freq2 = get_dominant_frequency(segment2, sr2)
            frequency_difference = abs(dominant_freq1 - dominant_freq2)
            frequency_tolerance = freq_tolerance_percent / 100.0 * abs(dominant_freq1)
            
            logging.debug(f"Segment Comparison - Max Amplitude1: {max_amplitude1}, Max Amplitude2: {max_amplitude2}")
            logging.debug(f"Segment Comparison - Amplitude Difference: {amplitude_difference}, Amplitude Tolerance: {amplitude_tolerance}")
            logging.debug(f"Segment Comparison - Frequency Difference: {frequency_difference}, Frequency Tolerance: {frequency_tolerance}")
            logging.debug(f"Segment Comparison - Dominant Frequency1: {dominant_freq1}, Dominant Frequency2: {dominant_freq2}")
            # print("flatten segment1, segment2: ", list(flatten(segment1)), list(flatten(segment2)))
            if np.round(amplitude_difference) <= np.round(amplitude_tolerance) and np.round(frequency_difference) <= np.round(frequency_tolerance):
                # distance, path = fastdtw(segment1.flatten(), segment2.flatten(), dist=euclidean)  # Ensure 1-D arrays
                distance, path = fastdtw(list(flatten(segment1)), list(flatten(segment2)), dist=euclidean)       
                logging.debug(f"Segment Comparison - DTW Distance: {distance}")
                if distance < (amplitude_tolerance + frequency_tolerance):
                    matched_segments.append((segment1, segment2))
                    sf.write(segment1_path, segment1, sr1)
                    sf.write(segment2_path, segment2, sr2)
                    return "Match", segment1_path, segment2_path

    if not matched_segments:
        return "No Match", None, None

    return "Partial Match", segment1_path, segment2_path

def get_dominant_frequency(segment, sr):
    logging.debug("Calculating dominant frequency.")
    fft = np.fft.fft(segment)
    freqs = np.fft.fftfreq(len(segment), 1/sr)
    magnitude = np.abs(fft)
    dominant_freq = freqs[np.argmax(magnitude)]
    return abs(dominant_freq)

def save_audio(audio, filepath):
    with open(filepath, 'wb') as f:
        f.write(audio.read())

@app.route('/compare', methods=['POST'])
def compare():
    if 'audio' not in request.files:
        logging.error("No audio file provided.")
        return jsonify({'result': 'No audio file provided'}), 400

    audio = request.files['audio']
    audio_type = request.form['type']

    if audio_type == 'master':
        audio_path = r"master.wav"
    elif audio_type == 'sample':
        audio_path = r"sample.wav"
    else:
        logging.error("Invalid audio type provided.")
        return jsonify({'result': 'Invalid audio type'}), 400

    save_audio(audio, audio_path)

    master_path = r"master.wav"
    sample_path = r"sample.wav"
    
    if os.path.exists(master_path) and os.path.exists(sample_path):
        result, segment1_path, segment2_path = compare_audios(master_path, sample_path)
        logging.debug(f"Comparison result: {result}")
        response = {'result': result}
        if segment1_path and segment2_path:
            response['master_segment'] = segment1_path
            response['sample_segment'] = segment2_path
        return jsonify(response)

    return jsonify({'result': 'Audio file saved'}), 200

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)



