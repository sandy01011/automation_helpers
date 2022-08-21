import bson
from pymongo import MongoClient
import os
from datetime import datetime



def dump(collections, conn, db_name, path):
    """
    MongoDB Dump

    :param collections: Database collections name
    :param conn: MongoDB client connection
    :param db_name: Database name
    :param path:
    :return:
    
    >>> DB_BACKUP_DIR = '/path/backups/'
    >>> conn = MongoClient("mongodb://admin:admin@127.0.0.1:27017", authSource="admin")
    >>> db_name = 'my_db'
    >>> collections = ['collection_name', 'collection_name1', 'collection_name2']
    >>> dump(collections, conn, db_name, DB_BACKUP_DIR)
    """

    #db = conn[db_name]
    db = conn
    print(f"printing conn type {type(conn)} and value {conn}")
    for coll in collections:
        with open(os.path.join(path, f'{db_name}_{coll}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.bson'), 'wb+') as f:
            for doc in db[coll].find():
                #print(f"doc type {type(doc)} doc value {doc}") 
                f.write(bson.BSON.encode(doc))


def restore(path, conn, db_name):
    """
    MongoDB Restore

    :param path: Database dumped path
    :param conn: MongoDB client connection
    :param db_name: Database name
    :return:
    
    >>> DB_BACKUP_DIR = '/path/backups/'
    >>> conn = MongoClient("mongodb://admin:admin@127.0.0.1:27017", authSource="admin")
    >>> db_name = 'my_db'
    >>> restore(DB_BACKUP_DIR, conn, db_name)
    
    """
    
    db = conn[db_name]
    for coll in os.listdir(path):
        if coll.endswith('.bson'):
            with open(os.path.join(path, coll), 'rb+') as f:
                db[coll.split('.')[0]].insert_many(bson.decode_all(f.read()))
