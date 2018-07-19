import numpy as np
import os
from sklearn.model_selection import train_test_split as tt_split 


def data_split(testprop): # enter a number between 0 and 1 for testprop, the remainder will be used for training
	cats = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '035', '036', '038', '039', '040', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '054', '055', '056', '057', '059', '060', '061', '062', '064', '065', '067', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '081', '082', '083', '084', '085', '086', '088', '090', '091', '093', '094', '095', '096', '097', '098', '099', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '157', '158', '159', '160', '162', '163', '164', '165', '166',  '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '181', '182', '183', '184', '185', '186', '187', '188', '192', '197', '198', '199', '200', '201', '202', '203', '205', '206', '207', '208', '210', '211', '212', '213', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230',  '232', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '275', '276', '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', '300', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '330', '331', '332', '333', '334', '335', '336', '337', '338', '339', '340', '341', '342']
	root_ref_dir = './ImitableFiles_Unfinished/' # the directory of directories contain reference sounds
	imi_dir = './imitationset_unorganized/' # the directory containing all imitation files

	train, test = tt_split(cats, test_size=testprop) 

	train_ref_dirs = [] # directories (aka categories) of reference files to be used for training
	train_imi_files = [] # filepaths of imitations files for training (corresponding to categories of reference files)
	test_ref_dirs = [] # directories (aka categories) of reference files to be used for testing
	test_imi_files = [] # filepaths of imitations files for testing (corresponding to categories of reference files)


	for refdirname, refdirnames, reffilenames in os.walk(root_ref_dir): # loop through categories
		for i in train: 
			if refdirname[27:30] == i[:3]:
				train_ref_dirs.append(refdirname+'/')
		for j in test: 
			if refdirname[27:30] == j[:3]:
				test_ref_dirs.append(refdirname+'/')
				

	for imidirname, imidirnames, imifilenames in os.walk(imi_dir):
		for imifilename in imifilenames:
			for i in train_ref_dirs:
				if imifilename[:3] == i[27:30]:
					train_imi_files.append(imidirname+imifilename)
			for j in test_ref_dirs:
				if imifilename[:3] == j[27:30]:
					test_imi_files.append(imidirname+imifilename)

	return train_ref_dirs, train_imi_files, test_ref_dirs, test_imi_files

def within_without(prop_in):
	


def main():
	train_ref, train_imi, test_ref, test_imi = data_split(.95)
	print test_ref[-1][27:30], test_imi[-1][27:30]


if __name__== "__main__":
    main()