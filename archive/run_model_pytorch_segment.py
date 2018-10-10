import pickle
import os
import numpy as np
import librosa
import torch
import csv
import torch.nn as nn
from siamese import *
from data_prep import *
from torch.utils.data import DataLoader
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
        if "DS_Store" not in f and ".js" not in f:
            
            
            y, sr = librosa.load(ref_dir+f, sr=44100)
            # zero-padding 

            y_copy = y

            if y_copy.shape[0] < 4*sr:
                pad = np.zeros((4*sr-y_copy.shape[0]))
                y_copy = np.append(y_copy, pad)

            # pad the audio file such that length is a multiple of 2 seconds (necessary if 4-second windows are 50% overlapped)
            if y_copy.shape[0]%(2*sr) != 0:
                pad = np.zeros((2*sr-y_copy.shape[0]%(2*sr)))
                y_copy = np.append(y_copy,pad)

            # print y_copy

            i = 0
            segments = []
            while ((i + (4*sr)) <= y_copy.shape[0]):
                segments.append(y_copy[i:(i+4*sr)])
                i = i + (2*sr)


            ref_spectrograms_segments = []

            for seg in segments:

                S = librosa.feature.melspectrogram(y=seg, sr=sr, n_fft=1024, hop_length=1024, power=2)

                S = S[:, 0:128]
                S_db = librosa.power_to_db(S, ref=np.max)
                ref_spec = [S_db]
                ref_spec = np.array(ref_spec).astype('float32')
                # print ref_spec.shape
                ref_spec = normalize_spectrogram(ref_spec)

                ref_spectrograms_segments.append(ref_spec)

            ref_file_names.append(f)

            ref_sepctrograms.append(ref_spectrograms_segments)
            ref_spectrograms_segments = []

    # print ref_file_names
    return np.array(ref_file_names), ref_sepctrograms

def preprocessing_imi(imi_path):

    # print imi_path
    y, sr = librosa.load(imi_path, sr=16000)
    # print 'y shape is: ',y.shape[0]

    y_copy = y

    if y_copy.shape[0] < 4*sr:
        pad = np.zeros((4*sr-y_copy.shape[0]))
        y_copy = np.append(y_copy, pad)

    # pad the audio file such that length is a multiple of 2 seconds (necessary if 4-second windows are 50% overlapped)
    if y_copy.shape[0]%(2*sr) != 0:
        pad = np.zeros((2*sr-y_copy.shape[0]%(2*sr)))
        y_copy = np.append(y_copy,pad)

    # print y_copy.shape[0]/(2*sr)

    i = 0
    segments = []
    while ((i + (4*sr)) <= y_copy.shape[0]):
        segments.append(y_copy[i:(i+4*sr)])
        i = i + (2*sr)

    imi_spectrogram_segments = []


    for seg in segments:
        S = librosa.feature.melspectrogram(y=seg, sr=sr, n_fft=133, 
                                       hop_length=133, power=2, n_mels=39, 
                                       fmin=0.0, fmax=5000)

        S = S[:, :482]
        S_db = librosa.power_to_db(S, ref=np.max)

        imi_spectrogram = [S_db]
    
        imi_spectrogram = np.array(imi_spectrogram).astype('float32')

        imi_spectrogram_norm = normalize_spectrogram(imi_spectrogram)

        imi_spectrogram_segments.append(imi_spectrogram_norm)

    return imi_spectrogram_segments

def load_model2(model, path):
    model.load_state_dict(torch.load(path, map_location = lambda storage, loc: storage))

def search_audio(imi_path, ref_dir, model):
    
    # Keras model     
    # model = load_model(model_path)
     # get architecture
     # get weights

    # imitation query
    imi_data = preprocessing_imi(imi_path)


    # To speed up, the spectrogram of reference recordings are saved.
    # If a new set of reference recordings are added, you should run 'preprecessing_ref' function 
    # ref_filenames, ref_data = preprocessing_ref(ref_dir)
    # np.save('./preprocessed_data1/ref_filenames.npy', ref_filenames)
    # np.save('./preprocessed_data1/ref_data.npy', ref_data)
    ref_filenames = np.load('./preprocessed_data1/ref_filenames.npy')
    ref_data = np.load('./preprocessed_data1/ref_data.npy')


    # create pairs of input to Siamese network

    avg_list = []
    mx_list = [] 
    # med_list = []

    for ref_segs, fname in zip(ref_data, ref_filenames):
        avg, mx = sim(imi_data, ref_segs, model)
        avg_list.append(avg)
        mx_list.append(mx)
        # med_list.append(med)

    sorted_avg_index = np.argsort(avg_list)[::-1]
    sorted_mx_index = np.argsort(mx_list)[::-1]
    # sorted_med_index = np.argsort(med_list)[::-1]


    # prediction
    # model_output = model.predict([pairs_imi_left, pairs_ref_right],  batch_size=1, verbose=1)
    
    # sort the reference recordings by the predicted similarities between the imitaiton and all the reference recodrings
    sorted_avg_filenames = ref_filenames[sorted_avg_index]
    sorted_mx_filenames = ref_filenames[sorted_mx_index]
    # sorted_med_filenames = ref_filenames[sorted_med_index]

    return sorted_avg_filenames, sorted_mx_filenames

def sim(X,Y, model): #takes segments of the imitation recording and segments of ONE reference recording, returns average similarity and max similarity
    left = []
    right = []
    for x in X:
        # for y in Y:
        #     left.append(x)
        #     right.append(y)
        dataset = InferenceData(x, Y)
        data = DataLoader(dataset, batch_size=128)

    # left = np.array(left)
    # right = np.array(right)

    # left = left.reshape(left.shape[0], 1, 39, 482)
    # right = right.reshape(right.shape[0], 1, 128, 128)



    outputs = np.array([])
    for imitation, reference in data: 
        # print imitation.shape
        # print reference.shape
        output = model(imitation, reference)  #   output will have shape 128
        outputs = np.concatenate([outputs, output.detach().cpu().numpy()])

    # model_output = model.predict([left, right],  batch_size=1, verbose=0)
    # model_output = model_output.flatten()
    #sorted_index = np.argsort(model_output.flatten())[::-1]

    avg = np.mean(outputs)
    mx = np.max(outputs)
    # med = np.median(outputs)

    return avg, mx

def main():
    imi_dir = './imitationset_unorganized/'
    ref_dir = './ImitableFiles_Unfinished/001Animal_Domestic animals_ pets_Cat_Growling/'
    model_path = './model/fine_tuned'
    siamese = Siamese()
    model = load_model2(siamese, model_path)
    # indices = []
    avg_indices = []
    mx_indices = []

    root_dir = './ImitableFiles_Unfinished/'
    allrefdirnames = []
    for refdirname, refdirnames, reffilenames in os.walk(root_dir):
        allrefdirnames.append(refdirname)

    allrefdirnames = allrefdirnames[1:]

    # preprocessing_ref(ref_dir)
    # print search_audio(imi_dir, ref_dir, model)

    # avg, mx = search_audio(imi_dir, ref_dir, model)
    # avg = list(avg)
    # mx = list(mx)
    # for i, data in enumerate(avg):
    #     if 'perfect' in data.lower():
    #         avg_indices.append(i)
    # for i, data in enumerate(mx):
    #     if 'perfect' in data.lower():
    #         mx_indices.append(i)

    # print 'rank of canonical using average similarity of segments: ', avg_indices[-1]+1
    # print 'rank of canonical using max similarity of segments: ', mx_indices[-1]+1

    with open('pytorch_ranks_finetuned_yesseg.csv', 'a') as output_file:
        csv_writer = csv.writer(output_file, delimiter=',')
        csv_writer.writerow(["Category", "Imitation file", "Search results using AVG", "Canonical rank using AVG", "Search results using MAX", "Rank of canonical using MAX of segment sims"])
        for refdirname, refdirnames, reffilenames in os.walk(root_dir): # loop through categories
            for imidirname, imidirnames, imifilenames in os.walk(imi_dir): # loop through imitations
                for rd, rdname in zip(refdirnames, allrefdirnames): # rd is used for matching ref audio category to imitation audio, rdname is used for skipping the root directory (outputted from os.walk)
                    ref_filenames, ref_data = preprocessing_ref(rdname+'/') # preprocess current reference audio category (doing this externally so it isn't done everytime search_audio is run)
                    np.save('./preprocessed_data1/ref_filenames.npy', ref_filenames)
                    np.save('./preprocessed_data1/ref_data.npy', ref_data)
                    for imifilename in imifilenames: # loop through imitation files
                        if imifilename[:3] == rd[:3]: # if the category of reference audio matches category of imitation file
                            imi_path = os.path.join(imidirname, imifilename) #get the filepath of the imitation file
                            if "DS_Store" not in imi_path: # check to avoid noBackendError in librosa by loading a .DS_Store file
                                print "Category: ", rd
                                print "Imitation file: ", imifilename
                                sorted_avg_filenames, sorted_mx_filenames = search_audio(imi_path, refdirname, siamese) # run the search on the current imitation file, current category, model
                                for i, elem in enumerate(sorted_avg_filenames): #search through the results for the canonical reference audio
                                    if 'perfect' in elem.lower(): # every canonical example has "perfect" in the title, so find the search result with 'perfect' in the filename
                                        avg_indices.append(i) # locate index (rank - 1) of the canonical reference audio file
                                for i, elem in enumerate(sorted_mx_filenames): #search through the results for the canonical reference audio
                                    if 'perfect' in elem.lower(): # every canonical example has "perfect" in the title, so find the search result with 'perfect' in the filename
                                        mx_indices.append(i) # locate index (rank - 1) of the canonical reference audio file
                                print "Rank of the canonical example (AVG SEG SIM): ", avg_indices[-1]+1, " out of ", len(sorted_avg_filenames), "\n"
                                print "Rank of the canonical example (MAX SEG SIM): ", mx_indices[-1]+1, " out of ", len(sorted_mx_filenames), "\n"
                                csv_writer.writerow([rd, imifilename, sorted_avg_filenames, avg_indices[-1]+1, sorted_mx_filenames, mx_indices[-1]+1], ) #write all the information to csv row
    
    


if __name__== "__main__":
    main()
