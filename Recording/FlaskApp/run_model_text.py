import pickle
import os
import numpy as np
import librosa
from keras.models import load_model
from keras import backend as K
import pickle
#K.set_image_dim_ordering('th')
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(1337)

old_ref_dir = './static/'

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
        if "DS_Store" not in f and ".js" not in f:
            print(f)
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

def get_old_ref_dir(ref_dir):
    try:
        with open('ref_dir.pickle', 'rb') as handle:
            old_ref_dir = pickle.load(handle)
    except EnvironmentError:
        with open('ref_dir.pickle', 'w+b') as handle:
            old_ref_dir = ref_dir
            pickle.dump(old_ref_dir, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("OLD_REF_DIR: ", old_ref_dir)
    return old_ref_dir

def update_old_ref_dir(old_ref_dir):
    with open('ref_dir.pickle', 'wb') as handle:
        pickle.dump(old_ref_dir, handle, protocol=pickle.HIGHEST_PROTOCOL)


def search_audio(imi_path, ref_dir, model, text):

    # imitation query
    imi_data = preprocessing_imi(imi_path)

    # check if given ref_dir was used last time. if it was, then use the saved list instead of re-preprocessing all ref files
    old_ref_dir = get_old_ref_dir(ref_dir)

    if ref_dir != old_ref_dir: # we need to preprocess the reference audio files if we haven't done that already
        print("REF_DIR updated. Preprocessing new reference data...")
        ref_filenames, ref_data = preprocessing_ref(ref_dir)
        np.save('./preprocessed_data/ref_filenames.npy', ref_filenames)
        np.save('./preprocessed_data/ref_data.npy', ref_data)

    old_ref_dir = ref_dir # save the current reference directory to check the next time we run the program
    update_old_ref_dir(old_ref_dir)

    old_ref_filenames = np.load('./preprocessed_data/ref_filenames.npy')
    old_ref_data = np.load('./preprocessed_data/ref_data.npy')

    i_delete = [] # contains indices of text-matched filenames (same indexing as spectrograms, so we can use these indices to remove those files from both the filenames and spectrgrams arrays)
    text_match_data = [] # will contain spectrograms of files who matched the text query
    text_match_filenames = [] # will contain filenames that matched the text query

    old_ref_filenames = [x.decode('utf-8') for x in old_ref_filenames]

    for i, fn in enumerate(old_ref_filenames): # gets all matches of text filenames (without extension) within our database and puts them in a list
        if text.lower() in fn[:-4].lower():
            i_delete.append(i)
            text_match_filenames.append(fn)
            text_match_data.append(old_ref_data[i])

    ref_filenames = np.delete(old_ref_filenames, i_delete,0) #remove the text matches from the original list of filenames
    ref_data = np.delete(old_ref_data, i_delete,0) #remove the matches from the original list of spectrograms
    sorted_filenames = inference(ref_data, ref_filenames, imi_data, model) #run the model on the non-matched files (i.e., sort them by similarity to imitation)

    if len(text_match_filenames) != 0: #if there was a text query supplied, run inference on the text matched files
        sorted_filenames_matched = inference(np.array(text_match_data), np.array(text_match_filenames), imi_data, model)
    else:
        sorted_filenames_matched = np.array([]) # if no text was supplied, there will be no matches, so return an empty array

    if text =='':
        temp = []
        temp = sorted_filenames_matched
        sorted_filenames_matched = sorted_filenames
        sorted_filenames = temp

    return sorted_filenames, sorted_filenames_matched


def inference(ref_data, ref_filenames, imi_data, model):
    # create pairs of input to Siamese network
    pairs_imi_left = []
    pairs_ref_right = []
    for ref in ref_data: # create an [imitation, reference] pair to go to the model for inference
        pairs_imi_left.append(imi_data[0])
        pairs_ref_right.append(ref)

    pairs_imi_left = np.array(pairs_imi_left)
    pairs_ref_right = np.array(pairs_ref_right)

    pairs_imi_left = pairs_imi_left.reshape(pairs_imi_left.shape[0], 1, 39, 482)
    pairs_ref_right = pairs_ref_right.reshape(pairs_ref_right.shape[0], 1, 128, 128)

    # prediction
    model_output = model.predict([pairs_imi_left, pairs_ref_right],  batch_size=1, verbose=1)
    model_output = np.array(model_output)


    # sort the reference recordings by the predicted similarities between the imitation and all the reference recodrings
    sorted_index = np.argsort(model_output.flatten())[::-1]
    sorted_filenames = ref_filenames[sorted_index]


    return sorted_filenames

def main():
    imi_path = './imitation/mosquito - 4938112476119040.wav'
    ref_dir = './static/'
    model_path = './model/model.h5'
    text = ''

    sorted_filenames, sorted_filenames_matched = search_audio(imi_path, ref_dir, model_path, text)
    print("files that matched your text description: ",sorted_filenames_matched)
    print("rest of files in directory sorted by similarity to vocal imitation: ", sorted_filenames)



if __name__== "__main__":
    main()
