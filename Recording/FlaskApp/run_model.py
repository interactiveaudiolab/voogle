import pickle
import os
import numpy as np
import librosa
from keras.models import load_model
from keras import backend as K
#K.set_image_dim_ordering('th')
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(1337)


def normalize_spectrogram(data):
    '''
    data: (num_examples, num_freq_bins, num_time_frames)
    '''
    m = np.mean(data, axis=(1,2))
    m = m.reshape(m.shape[0], 1, 1)
    std = np.std(data, axis=(1,2))
    std = std.reshape(std.shape[0], 1, 1)
    
    m_matrix = np.repeat(np.repeat(m, data.shape[1], axis=1), data.shape[2], axis=2)
    std_matrix = np.repeat(np.repeat(std, data.shape[1], axis=1), data.shape[2], axis=2)
    
    data_norm = np.multiply(data - m_matrix, 1./std_matrix)
    
    return data_norm


def preprocessing_ref(ref_dir):
    '''    
    ref_dir: path to the directory containing all the reference recording to search for

    '''

    file_list = os.listdir(ref_dir)

    ref_sepctrograms = []
    ref_file_names = []
    for f in file_list:
        print f
        ref_file_names.append(f)
        y, sr = librosa.load(ref_dir+f, sr=44100)
        # zero-padding 
        
        if y.shape[0] < 4*sr:
            pad = np.zeros((4*sr-y.shape[0]))
            y_fix = np.append(y, pad)
        else:
            y_fix = y[0:int(4*sr)]

        S = librosa.feature.melspectrogram(y=y_fix, sr=sr, n_fft=1024, hop_length=1024, power=2)

        S_db = librosa.power_to_db(S, ref=np.max)
        S_fix = S_db[:, 0:128]

        ref_sepctrograms.append(S_fix)

    ref_sepctrograms = np.array(ref_sepctrograms).astype('float32')

    ref_sepctrograms_norm = normalize_spectrogram(ref_sepctrograms)

    return np.array(ref_file_names), ref_sepctrograms_norm

def preprocessing_imi(imi_path):

    y, sr = librosa.load(imi_path, sr=16000)

    # zero-padding 
    if y.shape[0] < 4*sr:
        pad = np.zeros((4*sr-y.shape[0]))
        y_fix = np.append(y, pad)
    else:
        y_fix = y[0:int(4*sr)]

    S = librosa.feature.melspectrogram(y=y_fix, sr=sr, n_fft=133, 
                                       hop_length=133, power=2, n_mels=39, 
                                       fmin=0.0, fmax=5000)
    S = S[:, :482]
    S_db = librosa.power_to_db(S, ref=np.max)

    imi_spectrogram = [S_db]
    
    imi_spectrogram = np.array(imi_spectrogram).astype('float32')

    imi_spectrogram_norm = normalize_spectrogram(imi_spectrogram)

    return imi_spectrogram_norm


def search_audio(imi_path, ref_dir, model_path):
    
    # Keras model     
    model = load_model(model_path)

    # imitation query
    imi_data = preprocessing_imi(imi_path)


    # To speed up, the spectrogram of reference recordings are saved.
    # If a new set of reference recordings are added, you should run 'preprecessing_ref' function 
    # ref_filenames, ref_data = preprocessing_ref(ref_dir)
    # np.save('./preprocessed_data/ref_filenames.npy', ref_filenames)
    # np.save('./preprocessed_data/pairs_ref_right.npy', ref_data)
    ref_filenames = np.load('./preprocessed_data/ref_filenames.npy')
    ref_data = np.load('./preprocessed_data/ref_data.npy')


    # create pairs of input to Siamese network
    pairs_imi_left = []
    pairs_ref_right = []
    for ref in ref_data:
        pairs_imi_left.append(imi_data[0])
        pairs_ref_right.append(ref)

    pairs_imi_left = np.array(pairs_imi_left)
    pairs_ref_right = np.array(pairs_ref_right)

    pairs_imi_left = pairs_imi_left.reshape(pairs_imi_left.shape[0], 1, 39, 482)
    pairs_ref_right = pairs_ref_right.reshape(pairs_ref_right.shape[0], 1, 128, 128)

    # prediction
    model_output = model.predict([pairs_imi_left, pairs_ref_right],  batch_size=1, verbose=1)
    
    # sort the reference recordings by the predicted similarities between the imitaiton and all the reference recodrings
    sorted_index = np.argsort(model_output.flatten())[::-1]
    sorted_filenames = ref_filenames[sorted_index]
    
    return sorted_filenames

def main():
    imi_path = './imitation/mosquito - 4938112476119040.wav'
    ref_dir = './audio/'
    model_path = './model/model.h5'


    sorted_filenames = search_audio(imi_path, ref_dir, model_path)

    print sorted_filenames
    


if __name__== "__main__":
    main()
