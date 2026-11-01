# ruff: noqa: N999  (the cardReader folder name is camelCase by project convention)
"""The browser dashboard read.py --web serves in place of the window's sidebar.

    snapshot.py -> the reader's state for one frame as a plain dict, and the page's buttons
    server.py   -> serves page.html, streams snapshots to it as Server-Sent Events, and
                   queues the buttons pressed on it
    page.html   -> the dashboard: count, Cards Left grid, hands with hints, key buttons, status
"""
