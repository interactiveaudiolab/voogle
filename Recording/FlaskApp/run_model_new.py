import pickle
import os
import numpy as np
import librosa
from keras.models import load_model
from keras import backend as K
import csv
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
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
        if "DS_Store" not in f:
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
    # print imi_path
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
    this_model = load_model(model_path)

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
    pairs_imi_left = []
    pairs_ref_right = []
    for ref in ref_data:
        pairs_imi_left.append(imi_data[0])
        pairs_ref_right.append(ref)

    



    pairs_imi_left = np.array(pairs_imi_left)
    # print pairs_imi_left.shape


    pairs_ref_right = np.array(pairs_ref_right)
    # print pairs_ref_right.shape

    pairs_imi_left = pairs_imi_left.reshape(pairs_imi_left.shape[0], 1, 39, 482)
    pairs_ref_right = pairs_ref_right.reshape(pairs_ref_right.shape[0], 1, 128, 128)

    # prediction
    model_output = this_model.predict([pairs_imi_left, pairs_ref_right],  batch_size=1, verbose=1)
    
    # sort the reference recordings by the predicted similarities between the imitaiton and all the reference recodrings
    sorted_index = np.argsort(model_output.flatten())[::-1]
    sorted_filenames = ref_filenames[sorted_index]
    sorted_similarity= model_output.flatten()[sorted_index]
    
    return sorted_filenames, sorted_similarity

def main():
    # imi_dir = './imitationset_unorganized/'
    # ref_dir = './ImitableFiles_Unfinished/002Animal_Domestic animals_ pets_Cat_Hiss/'
    # model_path = './model/model.h5'

    imi_path = './silence_test_files/SONAR.WAV'
    ref_dir = './static/'
    model_path = './model/model_11-10_top_pair.h5'

    
    # indices = []
    # root_dir = './ImitableFiles_Unfinished/'
    # allrefdirnames = []
    # for refdirname, refdirnames, reffilenames in os.walk(root_dir):
    #     allrefdirnames.append(refdirname)

    # allrefdirnames = allrefdirnames[1:]
    
    # with open('results_keras_no_HNS_no_seg.csv', 'a') as output_file:
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
    #                             sorted_filenames, sorted_sims = search_audio(imi_path, refdirname, model) # run the search on the current imitation file, current category, model
    #                             for i, elem in enumerate(sorted_filenames): #search through the results for the canonical reference audio
    #                                 if 'perfect' in elem.lower(): # every canonical example has "perfect" in the title, so find the search result with 'perfect' in the filename
    #                                     indices.append(i) # locate index (rank - 1) of the canonical reference audio file
    #                             print "Rank of the canonical example: ", indices[-1]+1, " out of ", len(sorted_filenames), "\n"
    #                             csv_writer.writerow([rd, imifilename, sorted_filenames, indices[-1]+1], ) #write all the information to csv row
    
    # # ranks = [5,9,9,5,3,3,7,8,8,1,1,6,9,9,8,3,9,8,7,4,3,1,7,7,2,1,8,9,9,10,5,2,1,1,3,8,1,6,9,2,1,1,1,7,2,7,5,9,1,2,2,6,6,6,2,2,2,3,1,3,3,1,1,2,1,1,1,1,1,4,5,3,8,6,6,4,2,7,3,3,1,3,3,1,9,8,9,9,4,7,9,2,5,10,2,5,1,2,1,1,1,1,5,5,7,4,2,3,6,7,5,6,3,10,2,3,3,2,4,7,9,2,1,2,1,1,1,4,2,4,2,5,3,5,5,4,10,8,5,5,9,1,2,2,7,2,10,8,7,4,6,5,5,4,5,10,6,1,10,8,2,5,3,3,3,9,1,2,2,7,1,1,1,4,5,2,4,4,8,6,7,2,5,7,5,9,6,4,7,8,2,3,1,6,1,1,1,9,6,4,6,5,5,5,5,3,9,4]
    # # ranks = [1,4,9,9,7,10,4,5,2,6,1,2,6,5,5,9,10,1,5,8,1,9,1,9,5,10,5,8,7,3,8,5,5,1,3,5,3,2,1,6,9,6,5,6,1,6,7,6,8,7,1,4,3,2,3,5,2,3,8,2,1,3,4,10,1,10,9,6,1,1,3,6,9,3,2,3,9,3,1,5,7,8,1,3,10,1,9,5,5,7,4,6,3,4,3,2,9,6,6,2,6,6,10,6,6,9,6,1,8,1,9,3,1,3,9,8,7,6,3,1,5,2,1,3,5,10,9,1,1,5,10,5,6,2,3,10,6,1,8,3,1,1,6,3,1,5,10,4,3,5,1,8,1,4,7,4,1,9,5,7,10,3,5,8,8,9,2,4,5,9,1,6,5,6,10,3,4,2,4,9,6,3,2,7,8,1,4,2,5,9,9,10,6,1,6,7,9,9,4]
    # # bin_edges = [0,2,4,6,8,10]

    

    # n, bins, patches = plt.hist(ranks, bins=10, alpha=1)
    # plt.xticks(np.arange(min(ranks), max(ranks)+1, 1.0))
    # plt.xlabel('Rank (lower is better)')
    # plt.ylabel('Number of imitations with this rank')
    # plt.title('Rank of canonical examples in searches run on "worst" imitations')
    # plt.show()
    sorted_filenames, sorted_similarity = search_audio(imi_path, ref_dir, model_path)
    print sorted_filenames, sorted_similarity
    


if __name__== "__main__":
    main()
