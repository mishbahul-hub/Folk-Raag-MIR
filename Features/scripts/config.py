"""
config.py
---------
All tunable settings for the HMD feature extraction pipeline, in one place.
Import this module first — every other module depends on these values.
"""

# ===========================================================================
# Audio / feature settings (according to Thesis)
# ===========================================================================
SR = 16000            # sample rate every audio file gets resampled to
N_FFT = 512            # FFT window size
HOP_LENGTH = 320        # hop size between frames
N_MELS = 64             # number of mel bands (for log-mel spectrogram)
GT_WIN_TIME = 0.025     # gammatone window length in seconds (for cochleagram)
GT_HOP_TIME = 0.010     # gammatone hop length in seconds
GT_CHANNELS = 128       # number of gammatone filters (for cochleagram)
GT_F_MIN = 50           # lowest frequency the gammatone filterbank covers (Hz)

AUDIO_EXTS = (".mp3", ".wav")
UUID_LEN = 36            # a uuid is always 36 characters long

# ===========================================================================
# Paths — change these to match your Kaggle dataset location
# ===========================================================================
ROOT_DIR = "/kaggle/input/datasets/mishbhaul/indian-raga-dataset/RagaDataset/Hindustani/audio"
OUTPUT_DIR = "/kaggle/working/features_with_labels"

# ===========================================================================
# uuid (dataset folder name) -> raga name, for every raga we have audio for
# ===========================================================================
ID_TO_RAGA = {
    'f7fddfc0-8c1d-4dd2-90d5-5d51a99d61f8 2': 'Rāgēśrī',
    'f6432fec-e9c2-4b09-9e73-c46086cbd8ea': 'Ahira bhairav',
    '290674e0-d94c-41c1-ad99-f65fa22a1447': 'Madhuvanti',
    '0b3bbf97-0ec3-41da-add4-722d87329ec3': 'Madhukauns',
    'a9ee554f-f146-43e9-b2d0-9bb31fd4b57d': 'Kēdār',
    'd9c603fa-875f-4b84-b851-c6a345427898': 'Ābhōgī',
    '1e7de02f-e77f-405a-a033-f31117aaf955': 'Bhairav',
    '3eb7ba30-4b94-432e-9618-875ee57e01ab': 'Mārvā',
    '46997b02-f09c-4969-8138-4e1861f61967': 'Śrī',
    '54c4214c-05b9-4acc-8f77-6d5786e43a2e': 'Mārūbihāg',
    'a99e07d5-20a0-467b-8dcd-aa5a095177fd': 'Lalit',
    '62b79291-73b0-4c77-a353-0f8bc6ed8362': 'Dēś',
    '118401e7-8de8-4d81-9d8e-8070889e3fa8': 'Darbāri',
    '7591faad-e68a-4550-b675-8082842c6056': 'Bāgēśrī',
    '93c73081-bdf8-4eca-b325-d736b71e9b4b': 'Khamāj',
    'ba3242d0-8d40-4d93-865e-d2e2497ea2a8': 'Gauḍ malhār',
    'ecd04ceb-b46c-47fc-9045-84ac9160e527': 'Basant',
    '6f13484e-6fdd-402d-baf3-3835835454d0': 'Tōḍī',
    'fa28470c-d413-44c7-94da-181f530cbfdd': 'Puriyā dhanaśrī',
    'f7fddfc0-8c1d-4dd2-90d5-5d51a99d61f8': 'Rāgēśrī',
    '40dbe1db-858f-4366-bef8-9076eb67340d': 'Śuddh sāraṅg',
    '1b05a564-059f-445b-b325-cf26318367e3': 'Miyān malhār',
    'b143adaa-f1a6-4de4-8985-a5bd35e96279': 'Bairāgi',
    '64e5fb9e-5569-4e80-8e6c-f543af9469c7': 'Mālkauns',
    '2ed9379f-14c9-49af-8e4d-f9b63e96801f': 'Alahaiyā bilāval',
    'e771a74d-545d-41d5-816b-43403a818b0c': 'Bhūp',
    '063ea5a0-23b1-4bb5-8537-3d924fe8ebb3': 'Jōg',
    'a7d98897-e2fe-4d75-b9dc-e5b4dc88e1c6': 'Bihāg',
    '48b37bed-e847-4882-8a01-5c721e07f07d': 'Yaman kalyāṇ',
    '595de771-fcc2-414c-9df6-ae4ffd898549': 'Hamsadhvāni',
    'dd59147d-8775-44ff-a36b-0d9f15b31319': 'Bilāsakhānī tōḍī',
}
