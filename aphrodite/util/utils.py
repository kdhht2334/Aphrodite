import os
import json
import datetime


def count_files_in_directory(directory):
    file_count = len(
        [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    )
    return file_count


def delete_oldest_files(directory, num_to_keep):
    files = [
        (f, os.path.getmtime(os.path.join(directory, f)))
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]

    for i in range(min(num_to_keep, len(files))):
        file_to_delete = os.path.join(directory, files[i][0])
        os.remove(file_to_delete)


def delete_files_if_exceed_threshold(directory, threshold, num_to_keep):
    file_count = count_files_in_directory(directory)
    if file_count > threshold:
        delete_count = file_count - num_to_keep
        delete_oldest_files(directory, delete_count)


def save_logs(log_path, messages, response):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    delete_files_if_exceed_threshold(log_path, 20, 10)
    log_path = log_path if log_path else "logs"
    log = {}
    log["input"] = messages
    log["output"] = response
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(
        log_path, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".json"
    )
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
