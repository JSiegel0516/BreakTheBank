"""The card reader library.

Reading (read.py):
    camera.py    -> get a frame
    detector.py  -> find and name every corner index in the frame
    tracker.py   -> turn noisy per-frame detections into the ordered list of cards

Training (train.py):
    cardArt.py   -> crop and straighten photos of your deck into card art
    synthetic.py -> scatter card art on backgrounds to make labelled images
    dataset.py   -> combine labelled exports and synthetic images into data/dataset

deck.py holds the 53 labels everything shares.
"""
