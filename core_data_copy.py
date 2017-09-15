'''
先生からもらった細胞のデータフォルダに入ってる画像を一つのフォルダにまとめるプログラム
ファイルパスはDownload/cell_dataのように/で区切る
変な引数渡したときの処理とかはめんどくさいからあまりしてない
コピー成功時のメッセージのファイルパスが\と/が混じってたりするけどこれはwindowsとlinuxの違いだから挙動に影響はない
python core_data_copy.py {細胞データのフォルダパス} {移したい場所}
'''

import os
import sys
import shutil

def folder_copy(copyfrom, copyto):
    for filename in os.listdir(copyfrom):
        filepath = os.path.join(copyfrom, filename)
        if os.path.isdir(filepath):
            folder_copy(filepath, copyto)
        elif os.path.isfile(filepath):
            copypath = os.path.join(copyto, filename)
            shutil.copy(filepath, copypath)
            print('{0}から{1}にファイルをコピー'.format(filepath, copypath))

copyfrom = sys.argv[1]
copyto = sys.argv[2]
if not os.path.isdir(copyto):
    os.mkdir(copyto)
    print('{0}にフォルダを作成'.format(copyto))
folder_copy(copyfrom, copyto)
