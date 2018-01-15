import pandas as pd
import numpy as np
import datetime as dt

desired_width = 320
pd.set_option('display.width', desired_width)

data_path = pd.read_excel('data.xlsx', sheet_name='Пути')
data_sch = pd.read_excel('data.xlsx', sheet_name='Расписание')
data_work = pd.read_excel('data.xlsx', sheet_name='Работы')
data_joint = pd.read_excel('data.xlsx', sheet_name='Стыки')

data_path.columns = ['id_path', 'id_joint1', 'id_joint2', 'dist_path']
data_sch.columns = ['id_sch', 'id_train', 'id_joint_start', 'id_joint_finish',
                    'id_joints_sch', 'time_start', 'time_finish']
data_work.columns = ['id_work', 'id_train', 'id_joint_start',
                     'id_joint_finish', 'time_start', 'time_work']
data_joint.columns = ['id_joint', 'name_joint', 'cord_x', 'cord_y', 'type_joint']


# Преобразвоание пути из расписания в список из стыков
def trans_data_sch(path):
    return [int(joint) for joint in list(path) if joint != ',']


data_sch[['time_start', 'time_finish']] = data_sch[['time_start', 'time_finish']].apply(pd.to_datetime)
data_sch['id_joints_sch'] = data_sch['id_joints_sch'].apply(trans_data_sch)
data_work[['time_start', 'time_work']] = data_work[['time_start', 'time_work']].apply(pd.to_datetime)

# Добавление для каждого пути из joint1 в joint2 путь из joint2 в joint1
data_path_reverse = data_path.rename(columns={'id_joint1': 'id_joint2', 'id_joint2': 'id_joint1'})
data_path_reverse['id_path'] = data_path_reverse['id_path'] + data_path.shape[0]
data_path = data_path.append(data_path_reverse, ignore_index=True)

count_joint = data_joint.shape[0]
count_path = data_path.shape[0]
count_sch = data_sch.shape[0]
count_work = data_work.shape[0]


dist_sch_list = list()  # Вспомогательный список, в котором хранится длина всего пути для каждого задания пасс. поезда
pair_sch_list = list()  # Вспомогательный список, в котором храняться пары-(длина,путь) для каждого задания пасс. поезда


# Добавление в таблицу data_sch столбцов dist_sch и pair_joints_dist
for i in np.arange(count_sch):

    pair_joints_dist_list = list()
    buf_path = data_path[(data_sch['id_joint_start'][i] == data_path['id_joint1']) &
                         (data_sch['id_joints_sch'][i][0] == data_path['id_joint2'])]
    dist_path = buf_path['dist_path'].values[0]
    pair_joints_dist_list.append({'id_path': buf_path['id_path'].values[0], 'dist_path': dist_path})
    dist_sch = dist_path

    for j in np.arange(len(data_sch['id_joints_sch'][i]) - 1):

        buf_path = data_path[(data_sch['id_joints_sch'][i][j] == data_path['id_joint1']) &
                             (data_sch['id_joints_sch'][i][j + 1] == data_path['id_joint2'])]
        dist_path = buf_path['dist_path'].values[0]
        dist_sch += dist_path
        pair_joints_dist_list.append({'id_path': buf_path['id_path'].values[0], 'dist_path': dist_path})

    buf_path = data_path[(data_sch['id_joint_finish'][i] == data_path['id_joint2']) &
                         (data_sch['id_joints_sch'][i][len(data_sch['id_joints_sch'][i]) - 1] ==
                          data_path['id_joint1'])]
    dist_path = buf_path['dist_path'].values[0]
    pair_joints_dist_list.append({'id_path': buf_path['id_path'].values[0], 'dist_path': dist_path})
    dist_sch += dist_path

    dist_sch_list.append(dist_sch)
    pair_sch_list.append(pair_joints_dist_list)

data_sch['dist_sch'] = dist_sch_list
data_sch['pair_joints_dist'] = pair_sch_list


# Формирование таблицы А
tableA = pd.DataFrame(columns=['id_sch', 'id_path', 'time1', 'time2'])
start_datetime = np.datetime64(dt.datetime.combine(dt.date.today(), dt.time(0, 0, 0)))  # Начало моделирвоания


for i in np.arange(count_sch):

    time_start = data_sch['time_start'][i]
    time_finish = data_sch['time_finish'][i]
    time_delta = time_finish - time_start

    for j in np.arange(len(data_sch['pair_joints_dist'][i])):

        # Задание времени прохождения стыков пропорционально расстоянию между ними
        time_buf = time_start + time_delta * (data_sch['pair_joints_dist'][i][j]['dist_path'] / data_sch['dist_sch'][i])

        if j == 0:

            tableA.loc[len(tableA)] = [data_sch['id_sch'][i],
                                       data_sch['pair_joints_dist'][i][j]['id_path'],
                                       time_start, time_buf]

        elif j == len(data_sch['pair_joints_dist'][i])-1:

            tableA.loc[len(tableA)] = [data_sch['id_sch'][i],
                                       data_sch['pair_joints_dist'][i][j]['id_path'],
                                       time_buf, time_finish]

        else:

            tableA.loc[len(tableA)] = [data_sch['id_sch'][i],
                                       data_sch['pair_joints_dist'][i][j]['id_path'],
                                       tableA.loc[len(tableA)-1, 'time2'], time_buf]


# Формирование таблицы для выделения промежутков времени выполнения работ
table_time = pd.DataFrame(columns=['id_time', 'id_work', 'time_start', 'time_finish'])
start_work = start_datetime

for i in np.arange(count_work):

    if i == 0:
        table_time.loc[i] = [i+1, data_work['id_work'][i], start_work, data_work['time_start'][i]]
    else:

        time_start = table_time['time_finish'][i-1] + dt.timedelta(hours=data_work['time_work'][i-1].hour,
                                                                   minutes=data_work['time_work'][i - 1].minute,
                                                                   seconds=data_work['time_work'][i - 1].second)
        table_time.loc[i] = [i+1,
                             data_work['id_work'][i],
                             time_start,
                             data_work['time_start'][i]]


# Формирование таблицы B
tableB = pd.DataFrame(columns=['id_work', 'id_path', 'id_time', 'weight'])


for i, row in table_time.iterrows():

    buf = tableA[((tableA['time1'] >= row['time_start']) & (tableA['time1'] <= row['time_finish'])) |
                 ((tableA['time2'] >= row['time_start']) & (tableA['time2'] <= row['time_finish']))]

    data_weight = pd.DataFrame(np.vstack((buf['id_path'].value_counts().index,
                                          buf['id_path'].value_counts().values)).T, columns=['id_path', 'weight'])

    for j in np.arange(data_weight.shape[0]):

        tableB.loc[len(tableB)] = [row['id_work'], data_weight['id_path'][j], row['id_time'], data_weight['weight'][j]]


# Алгоритм Флойда-Уоршелла
def min_path_floyd(mtr_dist, mtr_path):

    for k in np.arange(count_joint):
        for i in np.arange(count_joint):
            for j in np.arange(count_joint):

                if mtr_dist[i, j] > mtr_dist[i, k] + mtr_dist[k, j]:
                    mtr_path[i, j] = mtr_path[i, k]

                mtr_dist[i, j] = min(mtr_dist[i, j], mtr_dist[i, k] + mtr_dist[k, j])

    return mtr_dist, mtr_path


# Нахождения минимального пути по mtr_graph_path
def min_path_joint(mtr_path, i, j):

    path = list()
    ind = i
    path.append(i)

    while ind != j:
        path.append(mtr_path[ind-1, j-1]+1)
        ind = mtr_path[ind-1, j-1] + 1

    return path


# Формироавние матриц для нахождения минимального пути
mtr_graph_dist = np.zeros((count_joint, count_joint), dtype=float)
mtr_graph_path = np.zeros((count_joint, count_joint), dtype=int)

for i in np.arange(count_path):
    mtr_graph_dist[data_path['id_joint1'][i] - 1, data_path['id_joint2'][i] - 1] = 1

for i in np.arange(count_joint):
    for j in np.arange(count_joint):

        if mtr_graph_dist[i, j] == 0:
            mtr_graph_dist[i, j] = float('inf')
        if i == j or mtr_graph_dist[i, j] != float('inf'):
            mtr_graph_path[i, j] = j
        else:
            mtr_graph_path[i, j] = -1


path_work_list = list()  # Список для хранения сформированных путей для каждой работы маневрого локомотива

for i in np.arange(count_work):

    buf = tableB[tableB['id_work'] == (i+1)]  # Пути с полученными весами для каждой работы

    # Формирование матрицы смежности, согласно весам
    for _, row in buf.iterrows():

         path_buf = data_path[data_path['id_path'] == row['id_path']]
         mtr_graph_dist[path_buf['id_joint1']-1, path_buf['id_joint2']-1] = row['weight'] + 1

    mtr_dist, mtr_path = min_path_floyd(mtr_graph_dist, mtr_graph_path)
    path = min_path_joint(mtr_path, data_work['id_joint_start'][i], data_work['id_joint_finish'][i])
    path_work_list.append({'id_work': data_work['id_work'][i], 'path': path})


print(path_work_list)



