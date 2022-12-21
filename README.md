## Basic Reinforcement Trader

This project attempts to learn to trade cryptocurrency successfully on its own. To achieve this, it uses the technique called reinforcement learning.

The main point is to have an **environment** that simulates trading activities and returns the corresponding rewards for the given actions. The other part, the **agent** receives the current state of the environment, the next state of the environment and the earlier mentioned rewards.

The agent stores a Q-table. In this project, it's a Deep Q Neural Network. The Q-table indexes the states and actions to a reward/score. It always selects the best action with the highest score for the best result. Random exploration is applied in train mode to avoid biases and support better generalization.

---

⚠️ _DISCLAIMER: This repository's only purpose is to showcase the author's skills. Trading is a high-risk activity. Use this software at your own risk. The author does NOT take any responsibility for your trading results._

---

### Usage

Create the basic configs from the examples.

```plaintext
cp .env.example .env && cp config.example.py config.py
```

Use docker compose for simplicity. First build the image.

```plaintext
docker compose build
```

Create a database to store the historical market prices.

```plaintext
docker compose up -d db
```

Download historical data. It can take up to a few minutes based on time range and timeframe.

```plaintext
docker compose up download
```

Train the Deep Q Network. It shouldn't take too long for a small dataset. (docker log might fail to show the train progress counter properly)

```plaintext
docker compose up train
```

Run backtest to ensure the validity of the trained network.

```plaintext
docker compose up backtest
```

Train and backtest plot their results. Copy them from the attached volume to the storage folder.

```plaintext
sh copy_results.sh
```

The plots should look like the ones in the images. [https://imgur.com/a/yHG6ZV3](https://imgur.com/a/yHG6ZV3)

### Conclusion

Basic, even advanced neural networks aren't sufficient for predicting or learning to trade any kind of market. The reason is because there might be trends and certain behaviors on the market, but it's very close to a random walk. The seemingly good result on the training set is just the result of overfitting the market's noise. The test epochs always show a straight noisy line in the plot. An excellent test result is usually an indicator of data leakage from the future.

### Custom usage

You may want to check `python -m data --help` and `python -m reinforcement --help` for customized usage.

```plaintext
python -m data download --help
usage: Data manager module download [-h] [-s SINCE] [-u UNTIL] [-t TIMEFRAME]
It downloads data from exchange.
options:
 -h, --help            show this help message and exit
 -s SINCE, --since SINCE
                       ISO format date like '2020-01-01T00:00:00'
 -u UNTIL, --until UNTIL
                       (Optional) ISO format date like '2020-01-01T00:00:00'
 -t TIMEFRAME, --timeframe TIMEFRAME
                       Like '30m' or '1h'
```

```plaintext
python -m reinforcement train --help
usage: reinforcement train [-h] [-e EPOCHS] [-t TIMEFRAME]
train the model
options:
 -h, --help            show this help message and exit
 -e EPOCHS, --epochs EPOCHS
                       number of peochs
 -t TIMEFRAME, --timeframe TIMEFRAME
                       Like '30m' or '1h'
```

```plaintext
usage: reinforcement backtest [-h] [-t TIMEFRAME]
backtest the model
options:
 -h, --help            show this help message and exit
 -t TIMEFRAME, --timeframe TIMEFRAME
                       Like '30m' or '1h'
```