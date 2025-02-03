import os
import requests


def get_all_files(folder_path):
    files = []
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        if os.path.isfile(entry_path):
            files.append(entry_path)
    files.sort()
    return files


def download_file(url, file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)


def get_first_file(directory_path):
    return_value = None
    try:
        entries = os.listdir(directory_path)
        entries.sort()
        for entry in entries:
            full_path = os.path.join(directory_path, entry)
            if os.path.isfile(full_path):
                return_value = os.path.join(directory_path, entry)
    except Exception as e:
        print(f"An error occurred: {e}")
    return return_value
