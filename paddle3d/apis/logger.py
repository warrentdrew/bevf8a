import time
import sys

class ProgressBar(object):
    """
    Progress bar printer
    """

    def __init__(self, logger, total_num, flush_interval: float = 0.5):
        self.logger = logger
        self.last_flush_time = time.time()
        self.flush_interval = flush_interval
        self.total_num = total_num
        self._end = False

    def update(self, curr_num: float):
        progress = float(curr_num / self.total_num)
        msg = "[{:<50}][{}/{}] {:.2f}%".format(
            "#" * int(progress * 50), curr_num, self.total_num, progress * 100
        )
        need_flush = (time.time() - self.last_flush_time) >= self.flush_interval

        if need_flush or self._end:
            with self.logger.use_terminator("\r"):
                self.logger.info(msg)
            self.last_flush_time = time.time()

        if self._end:
            self.logger.info("")
