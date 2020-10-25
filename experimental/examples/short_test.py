import progressbar
import time

progress = progressbar.ProgressBar(max_value=5)

for i in range(5):
    time.sleep(2)
    progress.update(1)
