from run_model_pytorch_segment import *

imi_path = './silence_test_files/imi_silence.wav'
ref_dir = './silence_test_files/'
model_path = './model/random_selection'

siamese = Siamese()
model = load_model2(siamese, model_path)

sorted_filenames = []

sorted_filenames = search_audio(imi_path, ref_dir, siamese)

print sorted_filenames