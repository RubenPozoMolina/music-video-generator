import os


def get_all_files(folder_path):
    files = []
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        if os.path.isfile(entry_path):
            files.append(entry_path)
    files.sort()
    return files
