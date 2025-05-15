from HateSelling.HateSelling import RunHateSell
from AgentSelling.AgentSelling import RunAgentSell

import multiprocessing
import schedule
import time

if __name__ == '__main__':
    zero_out = multiprocessing.Process(target=RunHateSell, args=())
    zero_out.start()

    schedule.every().hour.at(":00").do(RunAgentSell)

    while True:
        schedule.run_pending()
        time.sleep(30)