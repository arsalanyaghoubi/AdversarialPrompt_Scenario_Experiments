import pathlib
import logging

logger = logging.getLogger(__name__)



def process_and_append_to_csv(txt_file_path, output_csv_path):
    txt_file_path = pathlib.Path(txt_file_path)
    output_csv_path = pathlib.Path(output_csv_path)
    json_data = None
    if txt_file_path.is_file():
        with open(txt_file_path, 'r', encoding='utf-8-sig') as f:
            json_data = json.load(f)
    if json_data is None:
        logger.warning("Skipping empty or invalid file: %s", txt_file_path.name)
        return
    df_new = pd.DataFrame(json_data)
    if 'criteria_targeted' in df_new.columns:
        df_new['criteria_targeted'] = df_new['criteria_targeted'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x)
    if output_csv_path.exists():
        df_new.to_csv(output_csv_path, mode='a', index=False, header=False, encoding='utf-8-sig')
        logger.info("Appended %d items to existing %s", len(df_new), output_csv_path.name)
    else:
        df_new.to_csv(output_csv_path, mode='w', index=False, header=True, encoding='utf-8')
        logger.info("Created new file and saved data to %s", output_csv_path.name)


def encoder_(folder_link):
    folder_link = pathlib.Path(folder_link)
    for file in folder_link.iterdir():
        if file.is_file() and file.suffix == ".csv":
            logger.info("Deleting file: %s", file.name)
            file.unlink()
    target_csv_file = folder_link / f"{folder_link.name}.csv"
    for file in sorted(folder_link.rglob("*Results.txt")):
        logger.debug("*" * 40)
        logger.info("Processing %s", file.name)
        logger.debug("File directory: %s", file)
        logger.debug("Saving to: %s", target_csv_file.name)
        process_and_append_to_csv(file, target_csv_file)