import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import socket, struct


# def ip2int(ip):
#     # print(ip)
#     packedIP = socket.inet_aton(ip)
#     return struct.unpack("!L", packedIP)[0]

def ip2int(ip):
    try:
        packedIP = socket.inet_aton(ip)
        return struct.unpack("!L", packedIP)[0]
    except (socket.error, OSError):
        return np.nan

def anonymize_ip(ip_series):
        anonymized_ip = {}
        new_ip = 1
        result = []
        for ip in ip_series:
            if ip not in anonymized_ip:
                anonymized_ip[ip] = new_ip
                new_ip += 1
            result.append(anonymized_ip[ip])
        return result


def anonymize_port(port_series):
        anonymized_port = {}
        new_port = 1
        result = []
        for port in port_series:
            if port not in anonymized_port:
                anonymized_port[port] = new_port
                new_port += 1
            result.append(anonymized_port[port])
        return result

root_folder = "/scratch/Malware/CICAndMal/"

X_data = []
y_data = []

for root, dirs, files in os.walk(root_folder):
    for file in files:
        file_path = os.path.join(root,file)

        df = pd.read_csv(file_path)

        df.columns = df.columns.str.strip()

        # df.drop(columns=[" Label"], inplace = True)
        # df.drop(columns=["Flow ID", "Timestamp", "Label"], inplace = True)
        df.drop(columns=["Flow ID", "Label"], inplace = True)
        # df[' Source IP'] = df[' Source IP'].apply(ip_tqo_int)
        # df[' Destination IP'] = df[' Destination IP'].apply(ip_to_int)

        df['Direction'] = df['Source IP'].apply(lambda x: 1 if x in ['10.42.0.42', '10.42.0.211', '10.42.0.151'] else -1)
        # print(df['Source IP'], df['Direction'])

        # numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # df_numeric = df[numeric_cols]

        # df['Source IP'] = df['Source IP'].apply(ip2int)
        # df['Destination IP'] = df['Destination IP'].apply(ip2int)

        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        first_timestamp = df['Timestamp'].iloc[0]
        df['Timestamp'] = (df['Timestamp'] - first_timestamp).dt.total_seconds()
    
        df['Source IP'] = anonymize_ip(df['Source IP'])
        df['Destination IP'] = anonymize_ip(df['Destination IP'])


        df['Source Port'] = anonymize_port(df['Source Port'])
        df['Destination Port'] = anonymize_port(df['Destination Port'])
        
        # df['Timestamp'] = timestamp_dir(df['Timestamp'], df['Direction'])

        df['Timestamp'] = df.apply(lambda row: -row['Timestamp'] if row['Direction'] == -1 else row['Timestamp'], axis=1)
        
        df = df.sort_values(by='Timestamp') 
        df['IPD'] = df['Timestamp'].diff().fillna(0)

        # Optional
        df['IPD'] = df.apply(lambda row: -row['IPD'] if row['Direction'] == -1 else row['IPD'], axis=1)  

        # print(df)
        # label = df.iloc[-1][' Label']
        label = file_path.split('/')[7]

        # print(label)
        # y_data.append((label))
        # print(y_data)
        # df.drop(columns=[" Label"], inplace = True)

        # X_data.append(df.to_numpy())    
        # DataFrame의 각 행을 리스트로 변환하여 X에 추가
        for index, row in df.iterrows():

            if all(isinstance(item, (float, int)) for item in row):
            # try: 
                # for item in row:
                #     if not isinstance(item, (float, int)): 
                    X_data.append(row.tolist())
                    y_data.append(label)

        
        # delete Flow ID, Timestamp,
        # data = data.drop(columns=["Flow ID", " Timestamp", " Label"])

        # data = data.drop(columns=[" Label"], inplace=True)
        
        # X_data.append(data)
        # y_data.append(y)

print(np.shape(X_data))
print(np.shape(y_data))

print(X_data[10:30])
print(len(X_data[0]))


print(y_data[10:30])
# print(y_data[300:330])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_data)

print(y_encoded[10:30])
output = '/scratch/Malware/CICAndMal/processed_data'

X_data = np.array(X_data) 
y_data = np.array(y_encoded)

# print(X_data)

# np.save(os.path.join(output, 'X_data_inet_aton.npy'), X_data)
# np.save(os.path.join(output, 'y_data_inet_aton.npy'), y_data)


valid_indices = ~np.isnan(X_data).any(axis=1) & ~np.isinf(X_data).any(axis=1)
X_data_clean = X_data[valid_indices]
y_data_clean = y_data[valid_indices]

print("After cleaning:", X_data_clean.shape)
print("After cleaning:", y_data_clean.shape)


X_train, X_test, y_train, y_test = train_test_split(X_data_clean, y_data_clean, test_size=0.2, random_state=42, stratify=y_data_clean)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


np.save(os.path.join(output, 'X_train.npy'), X_train)
np.save(os.path.join(output, 'y_train.npy'), y_train)
np.save(os.path.join(output, 'X_test.npy'), X_test)
np.save(os.path.join(output, 'y_test.npy'), y_test)
np.save(os.path.join(output, 'X_valid.npy'), X_valid)
np.save(os.path.join(output, 'y_valid.npy'), y_valid)
