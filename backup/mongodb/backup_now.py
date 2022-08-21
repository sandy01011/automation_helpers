import pymongo
from mongodb_backup import *
import os
DB_BACKUP_DIR = 'data_backups/'
URI = 'mongodb://localhost:5017'
username = 'shunya'
password = '1m0rt@L'
#conn = MongoClient("mongodb://aicol:@!c0l@127.0.0.1:5017", authSource="aicol")
#conn = MongoClient("mongodb://aicol:@!c0l@127.0.0.1:27017", authSource="aicol")
col_name = 'GeMi'  # mongodb collection name
client = pymongo.MongoClient(URI)
conn = client[col_name]
conn.authenticate(username, password)
#collections = ['browser', 'chrome_export', 'chrome_export_stl_win' ]
#collections = ['browserCollector', 'browserCollector_0']
collections = ['MODA'  ,'bidsearch'  ,'caturl' ,'gemtest','gemtestPAR' ,'gemtestRAW','onbids' ,'parsedONbids','parsedONbids-0','parsedRAbids'  ,'parsedbids' ,'parsedcat','ppONbids' ,'ppONbids-0' ,'ppRAbids','rawONbids','rawONbids-0','rawRAbids','rawallbrands' ,'rawallcategories','rawallproducts','rawallservices','rawbidresultlists','rawcat']
print(f"backup dir list {os.listdir(path=DB_BACKUP_DIR)}")
dump(collections, conn, col_name, DB_BACKUP_DIR)
print(f"backup dir list {os.listdir(path=DB_BACKUP_DIR)}")

