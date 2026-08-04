
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")


def clean_uuid(raw_name: str) -> str:
    '''Normalize a folder name to its canonical uuid string.'''
    name = raw_name.strip()
    name = re.sub(r"\s*\(\d+\)$", "", name)   # trailing " (1)", " (2)"
    name = re.sub(r"\s+\d+$", "", name)       # trailing " 2", " 3"
    return name.lower()


def build_normalized_lookup(id_to_raga: dict) -> dict:
    '''Collapse duplicate-suffixed uuid keys into one clean lookup table.'''
    return {clean_uuid(k): v for k, v in id_to_raga.items()}


def find_audio_files_with_labels(root_dir: str, id_to_raga: dict) -> list[tuple[str, str, str]]:
    '''Walk `root_dir` and return (filepath, raga_label, artist) for every
    audio file whose parent uuid folder is present in `id_to_raga`.
    '''
    lookup = build_normalized_lookup(id_to_raga)
    items = []

    for uuid_folder in sorted(os.listdir(root_dir)):
        uuid_path = os.path.join(root_dir, uuid_folder)
        if not os.path.isdir(uuid_path):
            continue

        raga_label = lookup.get(clean_uuid(uuid_folder))
        if raga_label is None:
            print(f"WARNING: no raga mapping for uuid folder '{uuid_folder}', skipping")
            continue

        for dirpath, _, filenames in os.walk(uuid_path):
            rel_path = os.path.relpath(dirpath, uuid_path)
            artist = rel_path.split(os.sep)[0] if rel_path != "." else "unknown"

            for fname in filenames:
                if fname.lower().endswith(AUDIO_EXTENSIONS):
                    items.append((os.path.join(dirpath, fname), raga_label, artist))

    return items