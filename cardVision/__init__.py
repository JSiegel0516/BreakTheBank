"""The card-vision library. One module per stage of the pipeline:

    camera.py    -> get a frame
    (YOLO)       -> find corner indices in the frame          (runLive.py)
    zones.py     -> which seat is each detection in
    tracker.py   -> turn noisy per-frame detections into a stable table state
    blackjack.py -> score each seat's hand
    deck.py      -> the 53 labels everything else shares
"""
