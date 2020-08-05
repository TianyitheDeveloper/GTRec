"""
使用pandas处理KDD数据，操作包括将真值型数据进行max-min归一化，将类
别特征通过get_dummies进行onehot编码，只处理了三个类别，其他类同

2020年8月5日 10点34分
"""
import pandas as pd

def ReadData(filename = 'KDDTrain+.txt'):
    names = [
        'duration',
        'protocol_type',
        'service',
        'flag',
        'src_bytes',
        'dst_bytes',
        'land',
        'wrong_fragment',
        'urgent',
        'hot',
        'num_failed_logins',
        'logged_in',
        'num_compromised',
        'root_shell',
        'su_attempted',
        'num_root',
        'num_file_creations' ,
        'num_shells' ,
        'num_access_files' ,
        'num_outbound_cmds' ,
        'is_host_login',
        'is_guest_login',
        'count',
        'srv_count',
        'serror_rate',
        'srv_serror_rate',
        'rerror_rate',
        'srv_rerror_rate',
        'same_srv_rate',
        'diff_srv_rate',
        'srv_diff_host_rate',
        'dst_host_count',
        'dst_host_srv_count',
        'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate',
        'dst_host_srv_serror_rate',
        'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
        'class']
    return pd.read_table(filename, header=None, sep=',',names=names)

def DataPreprocess(data):
    pd_result = pd.DataFrame()
    # 处理真值（max-min归一化）
    pd_result['duration'] = (data['duration'] - data['duration'].min()) / (
                data['duration'].max() - data['duration'].min())
    # 处理类别
    pd_result = pd.concat([pd_result, pd.get_dummies(data['protocol_type'])], axis=1)
    pd_result = pd.concat([pd_result, pd.get_dummies(data['service'])], axis=1)
    pd_result = pd.concat([pd_result, pd.get_dummies(data['flag'])], axis=1)
    return pd_result


if __name__ == '__main__':
    data = ReadData()
    train_data = DataPreprocess(data)
    print(train_data)






