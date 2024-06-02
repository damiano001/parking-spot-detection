import pickle

mask_path = './masks/mask.png'
video_path = './samples/parking_video.mp4'

SPOT_EMPTY = True
SPOT_NOT_EMPTY = False

MODEL = pickle.load(open("./models/model.p", "rb"))