import os
import pandas as pd

dir_path = "/scratch/Malware/iot23"

# capture number list
malware_selection = {
    "mirai": [43],
    "kenjiro": [17],
    "gafgyt": [60],
    "okiru": [36],
    "hakai": [8],
    "ircbot": [39],
    "linux.hajime": [9],
    "muhstik": [3],
    "hideandseek": [1],
    # excluded torii, trojan
}

def get_columns_from_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#fields'):
                return line.strip().split()[1:]
    return None

def combine_file(dir_path, num_list, mal_name):
    mal_family_dir = os.path.join(dir_path, 'mal_family_capture')
    os.makedirs(mal_family_dir, exist_ok=True)

    combined_df_list = []

    for number in num_list:
        entry = f'CTU-IoT-Malware-Capture-{number}-1/'
        full_path = os.path.join(dir_path, 'mal', entry)
        labeled_path = os.path.join(full_path, 'bro', 'conn.log.labeled')

        if not os.path.exists(labeled_path):
            print(f"Error!! does not exist: {labeled_path}")
            continue

        try:
            columns = get_columns_from_file(labeled_path)
            if columns is None:
                print(f"Error!! could not find columns: {labeled_path}")
                continue
            df = pd.read_csv(labeled_path,
                         sep=r'\s+',
                         comment='#',
                         names=columns,
                         engine='python')
            df['Label'] = mal_name
            combined_df_list.append(df)
            print(f"read: {entry}")

        except Exception as e:
            print(f"Error!! occurred while processing {labeled_path}: {e}")

    if combined_df_list:
        combined_df = pd.concat(combined_df_list, ignore_index=True)
        save_path = os.path.join(mal_family_dir, f"{mal_name}.csv")
        combined_df.to_csv(save_path, index=False)
        print(f"Success!! {save_path}")

        label_counts = combined_df['Label'].value_counts()
        print(f"[Label distribution]\n{label_counts}")
    else:
        print(f"Warning!! No files to merge: {mal_name}")

# every malware family
for mal_name, captures in malware_selection.items():
    combine_file(dir_path, captures, mal_name)
