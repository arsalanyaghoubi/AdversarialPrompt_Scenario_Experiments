import pathlib
import json
import pandas as pd


def encoder_(folder_link):
    folder_link = pathlib.Path(folder_link)

    # 1. Clear out old CSV files first
    for file in folder_link.iterdir():
        if file.is_file() and file.suffix == ".csv":
            print(f"Deleting file: {file.name}")
            file.unlink()


    target_csv_file = ""
    for file in folder_link.iterdir():
        if file.is_file() and file.name.endswith("Results.txt"):
            if target_csv_file == "":
                target_csv_file = folder_link / f"{file.name.strip('.txt').replace('C1','').replace('C2','').replace('C3', '')}.csv"
            print(40*"*")
            print(f"Processing {file.name}")
            print(f"File directory: {file}")
            print(f"Saving to: {target_csv_file.name}")

            process_and_append_to_csv(file, target_csv_file)


def process_and_append_to_csv(txt_file_path, output_csv_path):
    txt_file_path = pathlib.Path(txt_file_path)
    output_csv_path = pathlib.Path(output_csv_path)

    json_data = None

    if txt_file_path.is_file():
        with open(txt_file_path, 'r', encoding='utf-8-sig') as f:
            json_data = json.load(f)

    if json_data is None:
        print(f"Skipping empty or invalid file: {txt_file_path.name}")
        return

    df_new = pd.DataFrame(json_data)

    if 'criteria_targeted' in df_new.columns:
        df_new['criteria_targeted'] = df_new['criteria_targeted'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x)

    if output_csv_path.exists():
        df_new.to_csv(output_csv_path, mode='a', index=False, header=False, encoding='utf-8-sig')
        print(f"Successfully appended {len(df_new)} items to existing {output_csv_path.name}\n")
    else:
        df_new.to_csv(output_csv_path, mode='w', index=False, header=True, encoding='utf-8')
        print(f"Created a brand new file and saved data to {output_csv_path.name}\n")


if __name__ == '__main__':
    pathdir = input("Please enter the directory path: \n")
    encoder_(pathdir)