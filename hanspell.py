# !pip install symspellpy-ko

import pandas as pd
import argparse
from symspellpy_ko import KoSymSpell, Verbosity


def main():
    train_df = pd.read_csv(args.read_path)

    sym_spell = KoSymSpell()
    sym_spell.load_korean_dictionary(decompose_korean=True, load_bigrams=True)

    s1_results = []
    s2_results = []

    string_list_1 = train_df['sentence_1'].tolist()
    string_list_2 = train_df['sentence_2'].tolist()


    for s in string_list_1:
        try:
            r = spell_checker.check(s)
            s1_results.append(r.checked)
        except Exception as e:
            s1_results.append(s)

    for s in string_list_2:
        try:
            r = spell_checker.check(s)
            s2_results.append(r.checked)
        except Exception as e:
            s2_results.append(s)

    # 리스트를 DataFrame에 열로 추가
    train_df['s1'] = s1_results
    train_df['s2'] = s2_results

    col = ['id','source','s1','s2','label','binary-label']
    df = train_df[col]
    df.columns = ['id','source','sentence_1','sentence_2','label','binary-label']


    df.to_csv(args.save_path, sep=",", index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', default='./train.csv', type=str)
    parser.add_argument('--save_path', default='./hanspell_train.csv', type=str)
    args = parser.parse_args()
    main() 