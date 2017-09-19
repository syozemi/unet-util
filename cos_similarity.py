def tfvec(wordvec):
    # 記事全体の単語数
    atwords = sum(wordvec.values())
    tfvec = {}
    for k, v in atwords.items:
        tfvec[k] = v * tf * idf
