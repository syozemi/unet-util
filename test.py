import mongo
import data_augmentation as da
import numpy as np

mg = mongo.Mongo()
image = np.array(mg.all_load({'teacher': 'valid'})[0]['original'])
# for i in range(2000):
#     print(mg.next_batch(20)[0]['_id'])
data = da.rotate(image, image, image)
print(len(data[0]))
# mg = mongo.Mongo()
# # for i in range(2000):
# #     print(mg.next_batch(20)[0]['_id'])
# t,x = mg.divide_data(0.8)
# print(len(t))
# print(len(x))
# print(t[-1]['_id'])
# print(x[0]['_id'])
# # t,x = mongo.load(0.8)
# # t.next_batch(20)
