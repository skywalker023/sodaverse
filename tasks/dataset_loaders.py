import os
import requests
import hashlib
import zipfile
import tarfile
import datetime
from tqdm import tqdm
import pandas as pd
import argparse
from datasets import load_dataset


class DownloadableFile:
    def __init__(self, url, filename, expected_hash, version="1.0", zipped=True):
        self.url = url
        self.filename = filename
        self.expected_hash = expected_hash
        self.zipped = zipped
        self.version = version


ATOMIC10X = DownloadableFile(
    url='https://storage.googleapis.com/ai2-mosaic-public/projects/soda/atomic10x_processed.tar.gz',
    filename='atomic10x_processed.tar.gz',
    expected_hash='445f815ca27717773e094e32fa292095e66deb375e9f77baa091f2f1d9282662',
    version="1.0",
    zipped=True
)

NAMES = DownloadableFile(
    url='https://storage.googleapis.com/ai2-mosaic-public/projects/soda/names.zip',
    filename='names.zip',
    expected_hash='60333f3266bbde0ea6bfd62f2720789e75ea8710099484f3b71f49446323c8a8',
    zipped=True
)

# =============================================================================================================

def unzip_file(file_path, directory='.'):
    if file_path.endswith(".zip"):
        target_location =  os.path.join(directory, os.path.splitext(os.path.basename(file_path))[0])
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(target_location)
    elif file_path.endswith(".tar.gz"):
        target_location =  os.path.join(directory, os.path.basename(file_path).split(".")[0])
        with tarfile.open(file_path) as tar:
            tar.extractall(target_location)

    return target_location

def check_built(path, version_string=None):
    """
    Check if '.built' flag has been set for that task.
    If a version_string is provided, this has to match, or the version is regarded as not built.
    """
    fname = os.path.join(path, '.built')
    if not os.path.isfile(fname):
        return False
    else:
        with open(fname, 'r') as read:
            text = read.read().split('\n')
        return len(text) > 1 and text[1] == version_string

def mark_built(path, version_string="1.0"):
    """
    Mark this path as prebuilt.
    Marks the path as done by adding a '.built' file with the current timestamp plus a version description string.
    """
    with open(os.path.join(path, '.built'), 'w') as write:
        write.write(str(datetime.datetime.today()))
        if version_string:
            write.write('\n' + version_string)

def download_and_check_hash(url, filename, expected_hash, version, directory='data', chunk_size=1024*1024*10):

    # Download the file
    response = requests.get(url, stream=True)
    try:
        total_size = int(response.headers.get('content-length', 0))
    except:
        print("Couldn't get content-length from response headers, using chunk_size instead")
        total_size = chunk_size
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    data = b''
    for chunk in response.iter_content(chunk_size=chunk_size):
        progress_bar.update(len(chunk))
        data += chunk
    progress_bar.close()

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the file to disk
    file_path = os.path.join(directory, filename)
    with open(file_path, 'wb') as f:
        f.write(data)

    # Calculate the hash of the downloaded data
    sha256_hash = hashlib.sha256(data).hexdigest()

    # Compare the calculated hash to the expected hash
    if sha256_hash != expected_hash:
        print('@@@ Downloaded file hash does not match expected hash!')
        raise RuntimeError

    return file_path

def build_data(resource, directory='data'):
    # check whether the file already exists
    if resource.filename.endswith('.tar.gz'):
        resource_dir = os.path.splitext(os.path.splitext(os.path.basename(resource.filename))[0])[0]
    else:
        resource_dir = os.path.splitext(os.path.basename(resource.filename))[0]
    file_path = os.path.join(directory, resource_dir)

    built = check_built(file_path, resource.version)

    if not built:
        # Download the file
        file_path = download_and_check_hash(resource.url, resource.filename, resource.expected_hash, resource.version, directory)

        # Unzip the file
        if resource.zipped:
            built_location = unzip_file(file_path, directory)
            # Delete the zip file
            os.remove(file_path)
        else:
            built_location = file_path

        mark_built(built_location, resource.version)
        print("Successfully built dataset at {}".format(built_location))
    else:
        print("Already built at {}. version {}".format(file_path, resource.version))
        built_location = file_path

    return built_location

def load(dataset, split='train'):
    if dataset.startswith("atomic10x"):
        dpath = build_data(ATOMIC10X)
        file = os.path.join(dpath, "ATOMIC10X_with_literals.parquet")
        df = pd.read_parquet(file)
    elif dataset == "names":
        dpath = build_data(NAMES)
        df = pd.read_csv(os.path.join(dpath, "names_1990-2021.csv"))
    elif dataset == "soda":
        dataset = load_dataset("allenai/soda", split=split)
        df = dataset.to_pandas()
    else:
        raise NotImplementedError

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arguments for dataset loaders')
    parser.add_argument('--dataset',
                        type=str,
                        help="Specify the dataset name.")
    parser.add_argument('--split',
                        type=str,
                        default="train",
                        help="Specify the split name. train, valid, test, etc.")
    args = parser.parse_args()
    load(args.dataset, args.split)