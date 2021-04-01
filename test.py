import time
from multiprocessing import JoinableQueue, Array
from libs.controller import Controller
from overwatch import Overwatch

if __name__ == "__main__":
    t = time.time()
    settings ={'symbol': 'SP500_M5_TA',
                'fraction': [1, 1e7],
                'window_size': 100,
                'num_workers': 24,
                'buffer_size': 100,
                'buffer_batch_size': 1,
                'sort_buffer': True,
                'skewed_memory': True,
                'shuffle_days': True,
                'normalization': False,
                'training_epochs': 2,
                'gamma': 0.99,
                'epsilon': [0.99, 0.999, 0.001],
                'lr_actor': 1e-5,
                'lr_critic': 1e-5,
                'verbose': 1,
                'start_time': t,
                }

    market = {'Spread': 1.2,
              'Risk': 0.1,
              'Slippage': 0.5,
              'MinLot': 0.1,
             }

    schedule = [1000, 50, 10]

    val = Array('i', [1 for _ in range(settings['num_workers'])])
    news_in_q = JoinableQueue()
    overwatch = Overwatch(in_q=news_in_q, val=val, settings=settings, schedule=schedule).start()
    controller = Controller(news_q=news_in_q, val=val, market=market, settings=settings)
    controller.work(schedule)
    controller.deinit()
    news_in_q.close()

