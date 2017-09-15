import pymongo
from pymongo.errors import BulkWriteError
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import process_data as pro
import cv2 as cv
#
# # mongodb へのアクセスを確立
# client = pymongo.MongoClient('localhost', 27017)
#
# # データベースを作成 (名前: my_database)
# db = client.my_database
#
# # コレクションを作成 (名前: my_collection)
# co = db.my_collection
#
# # なんか適当に保存
# co.insert_one({"test": 3})
#
# # 全部とってくる
# for data in co.find():
#     print(data)

# 教師用データを配列として返す
# typeは{'original': 'valid'}のような辞書でどのタイプの画像を返すかを指定する
class Mongo(object):
    def __init__(self):
        client = pymongo.MongoClient('localhost', 27017)
        db = client.cell_image
        self.train = db.train

    # prepareから呼び出せれる
    # bandの中に展開されたファイル群から画像の配列をデータベースに保存する
    def save(self):
        data = []

        # データが入ってるフォルダ
        folder = 'band/'
        for file in os.listdir(folder):
            if '.jpg' in file:
                image_path = folder + file
                cell_path = folder + file.replace('.jpg', '.mask.0.png')
                nucleus_path = folder + file.replace('.jpg', '.mask.1.png')

                #　後で下のに変える
                # original = processer.img_to_np(image_path)
                #画像を360*360の行列として取得する
                original = cv.imread(image_path,0)[3:,:] / 255
                # 正解データは388x388の形式にする
                cell, nucleus = [cv.resize(cv.imread(x)[3:,:,2]/255, (388,388)) for x in [cell_path,nucleus_path]]
                label = pro.create_mask_label(cell, nucleus)

                datum = {'original': original.tolist(), 'label': label.tolist()}
                self.train.insert_one(datum)
                # data.append(datum)

            else:
                pass

    def all_load(self):
        return list(self.train.find())
# 
# mongo = Mongo()
# mongo.save()
# data = mongo.all_load()
# print(data)
# print(type(data))
