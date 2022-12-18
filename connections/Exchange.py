import ccxt


class Exchange:
    def __enter__(self) -> ccxt.binance:
        self.client = ccxt.binance()
        return self.client

    def __exit__(self, *_) -> None:
        if self.client.session is not None:
            self.client.session.close()
