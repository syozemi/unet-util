# サンプル
# mongo = Mongo()
# band内の画像を処理してデータベースに保存
# mongo.prepare()
# 全画像取得
# data = mongo.all_load()
# print(data)
# print(type(data))
import pymongo
import numpy as np
import process_data as pro
import data_augmentation as da
import cv2 as cv

class Mongo(object):
    def __init__(self):
        client = pymongo.MongoClient('localhost', 27017)
        db = client.cell_image
        self.train = db.train
        self.index = 0
        self.train_data = []
        self.test_data = []
        self.train_index = 0
        self.test_index = 0

    # band内の画像を処理してデータベースに保存
    def prepare(self):
        if self.train.count() > 0:
            print('すでに{0}件画像が保存されています\n更新したい場合はデータベースを削除してください'.format(self.train.count()))
        else:
            print('実行中')
            self.save(setting)

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
                # 遅いからバルクインサートに変えたいけど一回しか実行しないからとりあえず放置
                self.train.insert_one(datum)

            else:
                pass
        print('{0}件の画像がデータベースに保存されました'.format(self.train.count()))

    # 全画像取得
    def all_load(self, setting):
        return list(self.train.find())

    # # 画像をデータベースからnum件ずつ取り出す
    # def next_batch(self, num):
    #     index = self.index
    #     length = self.train.count()
    #
    #     if self.index + num > length:
    #         array1 = list(self.train.find().skip(index))
    #         array2 = list(self.train.find().limit(num-(length-self.index)))
    #         x = array1 + array2
    #         self.index = num - (length-self.index)
    #         return x
    #     else:
    #         x = list(self.train.find().skip(index).limit(num))
    #         self.index = self.index + num
    #         return x
    #         # return list(self.train.find().batch_size(num))
    #
    def array_batch(self, num, array, index):
        length = len(array)
        if index + num > length:
            x = array[index:] + array[0:num-(length-index)]
            index = num - (length-index)
            return x
        else:
            x = array[index:index+num]
            index = index + num
            return x

    # num個教師用データを返す（divide_dataしておく必要あり)
    def next_train(self, num):
        return array_batch(num, self.train_data, self.train_index)

    # num個テストデータを返す（divide_dataしておく必要あり)
    def next_test(self, num):
        return array_batch(num, self.test_data, self.test_index)

    # すべての教師データを返す（divide_dataしておく必要あり)
    def all_train(self):
        return self.train_data

    # すべてのテストデータを返す（divide_dataしておく必要あり)
    def all_test(self):
        return self.test_data

    # rateの割合で教師用とテストデータを分ける
    def divide_data(self, rate, setting):
        # settingをprojectionに変換する（あとで）
        projection = setting
        data = np.random.permutation(list(self.train.find(projection=projection)))
        self.train_data = data[:int(self.train.count()*rate)]
        self.test_data = data[int(self.train.count()*rate):]
        return self.train_data, self.test_data
