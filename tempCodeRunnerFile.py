folder_data = "C:/FP Data Analys bangkit/Data"  

dfs = []
for name in os.listdir(folder_data):
    file_path = os.path.join(folder_data, name)
    df = pd.read_csv(file_path)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)