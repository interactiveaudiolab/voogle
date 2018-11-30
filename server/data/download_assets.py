import dropbox
import os
import zipfile
from log import get_logger

logger = get_logger('Dropbox')


def download_asset(asset, filepath):
    # Initialize dropbox only on first call to download_asset
    if not (hasattr(download_asset, 'dropbox')):
        download_asset.dropbox = dropbox.Dropbox(
            '9DlyeG0e-NoAAAAAAAAAOrLJxNBXR2PFswl78Ju6FpN5RAqj2RLk4BsdOI41MlFa')
    dbx = download_asset.dropbox

    logger.info('Downloading asset {}'.format(asset))

    # Download the file
    dbx.files_download_to_file(filepath, asset)


def download_dataset(filepath):
    # Make the dataset directory if it does not exist
    try:
        os.makedirs(filepath)
    except OSError:
        pass

    # Download the dataset zip file
    zip_filepath = filepath + '.zip'
    asset = '/data/' + os.path.basename(zip_filepath)
    download_asset(asset, zip_filepath)

    # Extract zip file
    logger.debug('Unzipping dataset file')
    with zipfile.ZipFile(zip_filepath) as zf:
        zf.extractall(os.path.dirname(filepath))

    # Remove zip file
    logger.debug('Removing zip file')
    os.remove(zip_filepath)


def download_model_weights(filepath):
    # Download the model weight file
    asset = '/model/' + os.path.basename(filepath)
    download_asset(asset, filepath)
