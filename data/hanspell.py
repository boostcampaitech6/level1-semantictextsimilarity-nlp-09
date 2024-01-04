# !pip install symspellpy-ko

import pandas as pd
import argparse
from hanspell import spell_checker    # !unzip hanspell.zip


def hanspell_apply() :
    train = pd.read_csv(args.read_path)

    s1_results = []
    s2_results = []

    string_list_1 = train['sentence_1'].tolist()
    string_list_2 = train['sentence_2'].tolist()
    # print(type(string))

    i=0
    for s in string_list_1:
        try:
            r = spell_checker.check(s)
            s1_results.append(r.checked)
            # if i%500 == 0 :
            #   print(i," ; ",s, " / ", r.checked)
        except Exception as e:
            s1_results.append(s)
        i += 1

    for s in string_list_2:
        try:
            r = spell_checker.check(s)
            s2_results.append(r.checked)
        except Exception as e:
            s2_results.append(s)

    # 리스트를 DataFrame에 열로 추가
    train['s1'] = s1_results
    train['s2'] = s2_results

    col = ['id','source','s1','s2','label','binary-label']
    train = train[col]
    train.columns = ['id','source','sentence_1','sentence_2','label','binary-label']


    train.to_csv(args.save_path, sep=",", index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', default='./data/train.csv', type=str)
    parser.add_argument('--save_path', default='./data/hanspell_train.csv', type=str)
    args = parser.parse_args()
    hanspell_apply()