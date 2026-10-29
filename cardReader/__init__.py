"""The card reader library.

Reading (read.py):
    camera.py    -> get a frame
    preprocess.py -> optional contrast help for a dim or washed-out frame
    detector.py  -> find and name every corner index in the frame
    tracker.py   -> turn noisy per-frame detections into the ordered list of cards
    table.py     -> which seat each card is in, and the hand at every seat
    reader.py    -> what happens per frame and per key press
    hud.py       -> the overlay
    cards.py holds the Detection, Card and Event types these pass around.

Training (train.py):
    cardArt.py   -> crop and straighten photos of your deck into card art
    synthetic.py -> scatter card art on backgrounds to make labelled images
    dataset.py   -> combine labelled exports and synthetic images into data/dataset

deck.py holds the 53 labels everything shares.
"""
