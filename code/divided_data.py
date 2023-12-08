import json
import random

min_log = 5

def fewer_than_min_log():
    with open('./data/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    # delete students who have fewer than min_log response logs
    stu_i = 0
    while stu_i < len(stus):
        if stus[stu_i]['log_num'] < min_log:

            del stus[stu_i]
            stu_i -= 1
        stu_i += 1
    print(len(stus))

def divide_data_5_5():
    with open('./5_5/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    train_slice, train_set, test_set = [], [], []
    for stu in stus:

        user_id = stu['user_id']
        stu_train = {'user_id': user_id}
        stu_test = {'user_id': user_id}
        train_size = int(stu['log_num'] * 0.5)
        test_size = stu['log_num'] - train_size
        logs = []
        for log in stu['log']:
            logs.append(log)
        random.shuffle(logs)
        stu_train['log_num'] = train_size
        stu_train['logs'] = logs[:train_size]
        stu_test['log_num'] = test_size
        stu_test['logs'] = logs[-test_size:]
        train_slice.append(stu_train)
        test_set.append(stu_test)
        # shuffle logs in train_slice together, get train_set
        for log in stu_train['logs']:
            train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                              'ques_content': log['ques_content'],
                              'img': log['img'], 'knowledge_code': log['knowledge_code']})
    random.shuffle(train_set)
    with open('5_5/train_slice.json', 'w', encoding='utf8') as output_file:
        json.dump(train_slice, output_file, indent=4, ensure_ascii=False)
    with open('5_5/train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open('5_5/test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)

def divide_data_6_4():
    with open('./6_4/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    train_slice, train_set, test_set = [], [], []
    for stu in stus:

        user_id = stu['user_id']
        stu_train = {'user_id': user_id}
        stu_test = {'user_id': user_id}
        train_size = int(stu['log_num'] * 0.6)
        test_size = stu['log_num'] - train_size
        logs = []
        for log in stu['log']:
            logs.append(log)
        random.shuffle(logs)
        stu_train['log_num'] = train_size
        stu_train['logs'] = logs[:train_size]
        stu_test['log_num'] = test_size
        stu_test['logs'] = logs[-test_size:]
        train_slice.append(stu_train)
        test_set.append(stu_test)
        # shuffle logs in train_slice together, get train_set
        for log in stu_train['logs']:
            train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                              'ques_content': log['ques_content'],
                              'img': log['img'], 'knowledge_code': log['knowledge_code']})
    random.shuffle(train_set)
    with open('6_4/train_slice.json', 'w', encoding='utf8') as output_file:
        json.dump(train_slice, output_file, indent=4, ensure_ascii=False)
    with open('6_4/train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open('6_4/test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)

def divide_data_7_3():
    with open('./7_3/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    train_slice, train_set, test_set = [], [], []
    for stu in stus:

        user_id = stu['user_id']
        stu_train = {'user_id': user_id}
        stu_test = {'user_id': user_id}
        train_size = int(stu['log_num'] * 0.7)
        test_size = stu['log_num'] - train_size
        logs = []
        for log in stu['log']:
            logs.append(log)
        random.shuffle(logs)
        stu_train['log_num'] = train_size
        stu_train['logs'] = logs[:train_size]
        stu_test['log_num'] = test_size
        stu_test['logs'] = logs[-test_size:]
        train_slice.append(stu_train)
        test_set.append(stu_test)
        # shuffle logs in train_slice together, get train_set
        for log in stu_train['logs']:
            train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                              'ques_content': log['ques_content'],
                              'img': log['img'], 'knowledge_code': log['knowledge_code']})
    random.shuffle(train_set)
    with open('7_3/train_slice.json', 'w', encoding='utf8') as output_file:
        json.dump(train_slice, output_file, indent=4, ensure_ascii=False)
    with open('7_3/train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open('7_3/test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)


def divide_data_8_2():
    with open('./8_2/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    train_slice, train_set, test_set = [], [], []
    for stu in stus:
        user_id = stu['user_id']
        stu_train = {'user_id': user_id}
        stu_test = {'user_id': user_id}
        train_size = int(stu['log_num'] * 0.9)
        test_size = stu['log_num'] - train_size
        logs = []
        for log in stu['log']:
            logs.append(log)
        random.shuffle(logs)
        stu_train['log_num'] = train_size
        stu_train['logs'] = logs[:train_size]
        stu_test['log_num'] = test_size
        stu_test['logs'] = logs[-test_size:]
        train_slice.append(stu_train)
        test_set.append(stu_test)
        # shuffle logs in train_slice together, get train_set
        for log in stu_train['logs']:
            train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                              'ques_content': log['ques_content'],
                              'img': log['img'], 'knowledge_code': log['knowledge_code']})
    random.shuffle(train_set)
    with open('9_1/train_slice.json', 'w', encoding='utf8') as output_file:
        json.dump(train_slice, output_file, indent=4, ensure_ascii=False)
    with open('9_1/train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open('9_1/test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)

divide_data_8_2()

def num():
    with open('./data/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    user = []
    know = []
    exer = []
    for i,item in enumerate(stus):
        if item['user_id'] not in user:
            user.append(item['user_id'])
        for j,jtem in enumerate(item['log']):
            if jtem['exer_id'] not in exer:
                exer.append(jtem['exer_id'])
            if jtem['knowledge_code'] not in know:
                know.append(jtem['knowledge_code'])
    print(len(user))
    print(len(know))
    print(len(exer))
