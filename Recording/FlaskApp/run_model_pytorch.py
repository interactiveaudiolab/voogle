import pickle
import os
import numpy as np
import librosa
import csv
import torch
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
            # print f
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

def load_model2(model, path):
    model.load_state_dict(torch.load(path, map_location = lambda storage, loc: storage))

def search_audio(imi_path, ref_dir, model):

    # imitation query
    imi_data = preprocessing_imi(imi_path)


    # To speed up, the spectrogram of reference recordings are saved.
    # If a new set of reference recordings are added, you should run 'preprocessing_ref' function 
    ref_filenames, ref_data = preprocessing_ref(ref_dir)
    np.save('./preprocessed_data1/ref_filenames.npy', ref_filenames)
    np.save('./preprocessed_data1/ref_data.npy', ref_data)
    ref_filenames = np.load('./preprocessed_data1/ref_filenames.npy')
    ref_data = np.load('./preprocessed_data1/ref_data.npy')


    # create pairs of input to Siamese network
    pairs_imi_left = []
    pairs_ref_right = []
    for ref in ref_data:
        pairs_imi_left.append(imi_data[0])
        pairs_ref_right.append(ref)

    # pairs_imi_left = np.array(pairs_imi_left)
    pairs_ref_right = np.array(pairs_ref_right)

    # pairs_imi_left = pairs_imi_left.reshape(pairs_imi_left.shape[0], 1, 39, 482)
    pairs_ref_right = pairs_ref_right.reshape(pairs_ref_right.shape[0], 1, 128, 128)


    # get the weights
    dataset = InferenceData(imi_data, pairs_ref_right)
    data = DataLoader(dataset, batch_size=128)

    outputs = np.array([])
    for imitation, reference in data: 
        output = model(imitation, reference)  #   output will have shape 128
        outputs = np.concatenate([outputs, output.detach().cpu().numpy()])
    

    
    # sort the reference recordings by the predicted similarities between the imitaiton and all the reference recodrings
    sorted_index = np.argsort(outputs)[::-1]
    sorted_filenames = ref_filenames[sorted_index]
    
    return sorted_filenames, outputs

def main():
    imi_path = './silence_test_files/SONAR.WAV'
    ref_dir = './static/'
    model_path = './model/fine_tuned'
    siamese = Siamese()
    model = load_model2(siamese, model_path)

    sorted_filenames, outputs = search_audio(imi_path, ref_dir, siamese)
    print sorted_filenames, outputs

    # imi_dir = './imitationset_unorganized/'
    # ref_dir = './ImitableFiles_Unfinished/001Animal_Domestic animals_ pets_Cat_Growling/'
    # model_path = './model/fine_tuned'
    # siamese = Siamese()
    # model = load_model2(siamese, model_path)

    # root_dir = './ImitableFiles_Unfinished/'
    # allrefdirnames = []
    # for refdirname, refdirnames, reffilenames in os.walk(root_dir):
    #     allrefdirnames.append(refdirname)

    # allrefdirnames = allrefdirnames[1:]

    # indices = []

    # # preprocessing_ref(ref_dir)
    # # print search_audio(imi_dir, ref_dir, model)

    # # avg, mx = search_audio(imi_dir, ref_dir, model)
    # # avg = list(avg)
    # # mx = list(mx)
    # # for i, data in enumerate(avg):
    # #     if 'perfect' in data.lower():
    # #         avg_indices.append(i)
    # # for i, data in enumerate(mx):
    # #     if 'perfect' in data.lower():
    # #         mx_indices.append(i)

    # # print 'rank of canonical using average similarity of segments: ', avg_indices[-1]+1
    # # print 'rank of canonical using max similarity of segments: ', mx_indices[-1]+1

    # with open('pytorch_ranks_finetuned_noseg.csv', 'a') as output_file:
    #     csv_writer = csv.writer(output_file, delimiter=',')
    #     csv_writer.writerow(["Category", "Imitation file", "Search results using AVG", "Canonical rank using AVG", "Search results using MAX", "Rank of canonical using MAX of segment sims"])
    #     for refdirname, refdirnames, reffilenames in os.walk(root_dir): # loop through categories
    #         for imidirname, imidirnames, imifilenames in os.walk(imi_dir): # loop through imitations
    #             for rd, rdname in zip(refdirnames, allrefdirnames): # rd is used for matching ref audio category to imitation audio, rdname is used for skipping the root directory (outputted from os.walk)
    #                 ref_filenames, ref_data = preprocessing_ref(rdname+'/') # preprocess current reference audio category (doing this externally so it isn't done everytime search_audio is run)
    #                 np.save('./preprocessed_data1/ref_filenames.npy', ref_filenames)
    #                 np.save('./preprocessed_data1/ref_data.npy', ref_data)
    #                 for imifilename in imifilenames: # loop through imitation files
    #                     if imifilename[:3] == rd[:3]: # if the category of reference audio matches category of imitation file
    #                         imi_path = os.path.join(imidirname, imifilename) #get the filepath of the imitation file
    #                         if "DS_Store" not in imi_path: # check to avoid noBackendError in librosa by loading a .DS_Store file
    #                             print "Category: ", rd
    #                             print "Imitation file: ", imifilename
    #                             sorted_filenames = search_audio(imi_path, refdirname, siamese) # run the search on the current imitation file, current category, model
    #                             for i, elem in enumerate(sorted_filenames): #search through the results for the canonical reference audio
    #                                 if 'perfect' in elem.lower(): # every canonical example has "perfect" in the title, so find the search result with 'perfect' in the filename
    #                                     indices.append(i) # locate index (rank - 1) of the canonical reference audio file
    #                             print "Rank of the canonical example: ", indices[-1]+1, " out of ", len(sorted_filenames), "\n"
    #                             csv_writer.writerow([rd, imifilename, sorted_filenames, indices[-1]+1], ) #write all the information to csv row
    
    
    


if __name__== "__main__":
    main()
