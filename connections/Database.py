import os
import dotenv

from pymongo import MongoClient


class Database:
    def __init__(self):
        dotenv.load_dotenv()

    def __enter__(self):
        self.client = MongoClient(
            host=os.environ["DB_HOST"], port=int(os.environ["DB_PORT"])
        )
        return self.client["brt"]

    def __exit__(self, *_) -> None:
        self.client.close()
