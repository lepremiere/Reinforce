import time
from multiprocessing import JoinableQueue, Array
from libs.controller import Controller
from overwatch import Overwatch

if __name__ == "__main__":
    t = time.time()
    settings ={'symbol': 'SP500_M1_TA',
                'fraction': [1e5, 1e5],
                'window_size': 100,
                'num_workers': 20,
                'buffer_size': int(1e3),
                'buffer_batch_size': 1,
                'normalization': False,
                'skewed': True,
                'training_epochs': 1,
                'lr_actor': 1e-5,
                'lr_critic': 2e-5,
                'verbose': 1,
                'start_time': t}
    market = {'Spread': 1.2,
              'Risk': 0.1,
              'Slippage': 0.5,
              'MinLot': 0.1,
             }

    cycles = 1000
    schedule = [1, 1]

    val = Array('i', [1 for _ in range(settings['num_workers'])])
    news_in_q = JoinableQueue()
    overwatch = Overwatch(in_q=news_in_q, val=val, settings=settings).start()
    controller = Controller(news_q=news_in_q, val=val, market=market, settings=settings)
    controller.work(cycles, schedule)
    controller.deinit()
    news_in_q.close()

